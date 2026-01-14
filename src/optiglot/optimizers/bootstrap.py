import hashlib
import json
import logging
import random
import threading
from typing import Any

import tqdm
from models import (
    RPCLM,
    RPCModule,
)

from .teleprompt import RPCTeleprompter
from .vanilla import LabeledFewShot

logger = logging.getLogger(__name__)


class BootstrapFewShot(RPCTeleprompter):
    """RPC version of BootstrapFewShot that delegates program execution to the server.

    Instead of running the teacher program locally with DSPy, this implementation:
    1. Sends rollout requests to the server with trace=True
    2. Server executes the program in its native language
    3. Collects successful traces from server responses
    4. Compiles student with bootstrapped demos
    """

    def __init__(
        self,
        metric_threshold: float | None = None,
        teacher_settings: dict | None = None,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 16,
        max_rounds: int = 1,
        max_errors: int | None = None,
        teacher_lm: RPCLM | None = None,
        evaluator: Any | None = None,
    ) -> None:
        """Initialize BootstrapFewShotRPC optimizer.

        Args:
            metric_threshold: If provided, only accept demos where score >= threshold.
                If None, any non-zero score is considered success.
            teacher_settings: Optional settings for the teacher model.
            max_bootstrapped_demos: Maximum number of bootstrapped demos per predictor.
            max_labeled_demos: Maximum number of labeled demos per predictor.
            max_rounds: Number of attempts per example to generate successful traces.
            max_errors: Maximum number of errors allowed during bootstrapping.
            teacher_lm: Optional LM config for teacher. If provided, overrides teacher's LM settings.
            evaluator: Optional evaluator instance.

        """
        super().__init__(evaluator=evaluator)
        self.metric_threshold = metric_threshold
        self.teacher_settings = {} if teacher_settings is None else teacher_settings
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.max_rounds = max_rounds
        self.max_errors = max_errors
        self.teacher_lm = teacher_lm
        self.error_count = 0
        self.error_lock = threading.Lock()

    def compile(
        self,
        student: RPCModule,
        *,
        trainset: list[dict[str, Any]],
        valset: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> RPCModule:
        """Compile student program using bootstrapped demonstrations.

        Args:
            student: The student program to compile.
            trainset: Training examples to bootstrap from.
            valset: Optional validation set.
            **kwargs: Additional keyword arguments.

        Returns:
            Compiled student program with bootstrapped and labeled demos.

        """
        self.trainset = trainset
        self.valset = valset

        self._prepare_student_and_teacher(student)
        self._prepare_predictor_mappings()
        self._bootstrap()

        self.student = self._train()

        return self.student

    def _prepare_student_and_teacher(
        self,
        student: RPCModule,
    ) -> None:
        self.student = student.reset_copy()
        self.teacher = student.deepcopy()

        if self.max_labeled_demos > 0:
            teleprompter = LabeledFewShot(k=self.max_labeled_demos)
            self.teacher = teleprompter.compile(
                self.teacher.reset_copy(),
                trainset=self.trainset,
            )

    def _prepare_predictor_mappings(self) -> None:
        name2predictor, predictor2name = {}, {}
        student, teacher = self.student, self.teacher

        assert len(student.predictors) == len(
            teacher.predictors,
        ), "Student and teacher must have the same number of predictors."

        for (name1, predictor1), (name2, predictor2) in zip(
            student.named_predictors.items(),
            teacher.named_predictors.items(),
            strict=False,
        ):
            assert name1 == name2, (
                "Student and teacher must have the same program structure."
            )
            assert predictor1.signature == predictor2.signature, (
                f"Student and teacher must have the same signatures for {name1}."
            )

            name2predictor[name1] = None
            # In RPC version, we map based on the predictor name directly
            # since the server returns names in the trace.
            # But we keep this for parity if needed.
            predictor2name[name1] = name1

        self.name2predictor = name2predictor
        self.predictor2name = predictor2name

    def _bootstrap(self) -> None:
        """Bootstrap demonstrations by running teacher on training examples via RPC."""
        max_bootstraps = self.max_bootstrapped_demos
        bootstrap_attempts = 0

        bootstrapped = {}
        # Map predictor names to collected demos
        self.name2traces: dict[str, list[dict[str, Any]]] = {
            name: [] for name in self.name2predictor
        }

        for example_idx, example in enumerate(tqdm.tqdm(self.trainset)):
            if len(bootstrapped) >= max_bootstraps:
                break

            # Try multiple rounds to get a successful trace
            for round_idx in range(self.max_rounds):
                bootstrap_attempts += 1
                if self._bootstrap_one_example(example, round_idx):
                    bootstrapped[example_idx] = True
                    break

        # Collect validation set (unbootstrapped examples)
        self.validation = [
            x for idx, x in enumerate(self.trainset) if idx not in bootstrapped
        ]
        random.Random(0).shuffle(self.validation)

        print(
            f"Bootstrapped {len(bootstrapped)} full traces after {example_idx + 1} examples "
            f"for up to {self.max_rounds} rounds, amounting to {bootstrap_attempts} attempts.",
        )

    def _bootstrap_one_example(
        self,
        example: dict[str, Any],
        round_idx: int = 0,
    ) -> bool:
        """Attempt to bootstrap a single example by calling the server.

        Args:
            example: The training example to bootstrap.
            round_idx: Current round number (for varying temperature/randomness on server).

        Returns:
            True if bootstrapping succeeded (metric passed), False otherwise.

        """
        # Remove this example from teacher's demos to avoid leakage
        teacher_copy = self.teacher.deepcopy()
        for predictor in teacher_copy.predictors:
            predictor.demos = [demo for demo in predictor.demos if demo != example]

        try:
            # Prepare LM for this round
            lm = self.teacher_lm
            if round_idx > 0:
                if lm:
                    lm = lm.model_copy(
                        update={"rollout_id": round_idx, "temperature": 1.0},
                    )
                else:
                    # If no teacher_lm provided, we can't easily bypass server cache
                    # without more info, but we can send a default one if needed.
                    pass

            # Call server to execute teacher on this example with tracing enabled
            response = self.evaluator.evaluate(
                canidates=[teacher_copy],
                devset=[example],
                trace=True,
                lm=lm,
            )

            # Should have exactly one result
            if not response.results or len(response.results) != 1:
                logger.warning(f"Expected 1 result, got {len(response.results)}")
                return False

            result = response.results[0]

            # Check if the prediction was successful based on metric score
            if result.score is None:
                logger.warning("No score returned from server")
                return False

            # Determine success
            if self.metric_threshold is not None:
                success = result.score >= self.metric_threshold
            else:
                # If no threshold, any positive score is success
                success = result.score > 0

            if not success:
                logger.debug(f"Example failed metric: score={result.score}")
                return False

            # Extract traces and add to our collection
            if not result.trace:
                logger.warning("No trace returned despite trace=True")
                return False

            # Group trace steps by predictor name
            name2demos: dict[str, list[dict[str, Any]]] = {}
            for step in result.trace:
                predictor_name = step.predictor_name

                # Create a demo from the step's inputs and prediction
                if step.prediction.output:
                    demo = {
                        "inputs": step.inputs,
                        "outputs": step.prediction.output,
                    }
                    name2demos.setdefault(predictor_name, []).append(demo)

            # Add collected demos to our traces
            for predictor_name, demos in name2demos.items():
                # If multiple traces for same predictor in one example,
                # sample 50/50 from first N-1 or last (like original DSPy)
                if len(demos) > 1:
                    # Use a stable hash for the seed
                    demo_str = json.dumps(
                        demos,
                        sort_keys=True,
                    )
                    seed = int(hashlib.md5(demo_str.encode()).hexdigest(), 16)
                    rng = random.Random(seed)
                    selected_demo = (
                        rng.choice(demos[:-1]) if rng.random() < 0.5 else demos[-1]
                    )
                    self.name2traces[predictor_name].append(selected_demo)
                else:
                    self.name2traces[predictor_name].extend(demos)

            return True

        except Exception as e:
            logger.error(f"Error bootstrapping example: {e}")
            with self.error_lock:
                self.error_count += 1
                current_error_count = self.error_count

            # Use max_errors if provided, otherwise assume a large default or re-raise
            if self.max_errors is not None and current_error_count >= self.max_errors:
                raise e
            return False

    def _train(self) -> RPCModule:
        """Train student by adding bootstrapped and labeled demos to each predictor."""
        rng = random.Random(0)
        raw_demos = self.validation

        for name, predictor in self.student.named_predictors.items():
            # Get bootstrapped demos for this predictor
            augmented_demos = self.name2traces.get(name, [])[
                : self.max_bootstrapped_demos
            ]

            # Fill remaining slots with labeled demos from validation set
            remaining_slots = self.max_labeled_demos - len(augmented_demos)
            sample_size = min(remaining_slots, len(raw_demos))
            sample_size = max(0, sample_size)

            # NOTE: Matching DSPy's behavior where raw_demos is reduced in each iteration
            raw_demos = rng.sample(raw_demos, sample_size)

            # Set predictor demos
            predictor.demos = augmented_demos + raw_demos

            logger.info(
                f"Predictor '{name}': {len(augmented_demos)} bootstrapped + "
                f"{len(raw_demos)} labeled demos",
            )

        return self.student
