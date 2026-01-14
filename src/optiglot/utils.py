import random
from logging import getLogger
from typing import TYPE_CHECKING, Any, Optional

import requests
from models import (
    RPCLM,
    RPCModule,
    RPCRolloutRequest,
    RPCRolloutResponse,
)
from settings import settings

if TYPE_CHECKING:
    import dspy


logger = getLogger(__name__)


class Evaluator:
    dspy_module: "dspy.Module" | None = None
    dspy_metric: Any | None = None

    def __init__(self):
        pass

    def set_dspy_module(self, dspy_module: "dspy.Module"):
        """Set the DSPy module to use for evaluation."""
        self.dspy_module = dspy_module

    def set_dspy_metric(self, dspy_metric: Any):
        """Set the DSPy metric to use for evaluation."""
        self.dspy_metric = dspy_metric

    def evaluate(
        self,
        canidates: list[RPCModule] | RPCModule,
        devset: list[dict[str, Any]],
        trace: bool = False,
        lm: RPCLM | None = None,
    ) -> RPCRolloutResponse:
        """Evaluate a list of programs on a list of examples.

        Args:
            canidates: The programs to evaluate.
            devset: The examples to evaluate the programs on.
            trace: Whether to trace the rollout.
            lm: The language model to use for the rollout.

        """
        # Check if we should use local DSPy evaluation

        dspy_module = self.dspy_module or settings.dspy_module

        if dspy_module is not None:
            from harnesses.dspy.dspy_harness import evaluate as dspy_evaluate

            # If we have a local DSPy module, we use the harness
            return dspy_evaluate(
                module=dspy_module,
                canidates=canidates,
                devset=devset,
                metric=self.dspy_metric,
                trace=trace,
                lm=lm,
            )

        if isinstance(canidates, RPCModule):
            canidates = [canidates]
        request = RPCRolloutRequest(
            canidates=canidates,
            examples=devset,
            trace=trace,
            lm=lm,
        )

        response = requests.post(
            f"{settings.host_url}/rollout",
            json=request.model_dump(),
        )
        response.raise_for_status()
        return RPCRolloutResponse.model_validate(response.json())


def create_minibatch(
    trainset: list[dict[str, Any]],
    batch_size: int = 50,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    """Create a minibatch from the trainset."""
    # Ensure batch_size isn't larger than the size of the dataset
    batch_size = min(batch_size, len(trainset))

    # If no RNG is provided, fall back to the global random instance
    rng = rng or random

    # Randomly sample indices for the mini-batch using the provided rng
    sampled_indices = rng.sample(range(len(trainset)), batch_size)

    # Create the mini-batch using the sampled indices
    minibatch = [trainset[i] for i in sampled_indices]

    return minibatch


def eval_candidate_program(
    batch_size: int,
    trainset: list[dict[str, Any]],
    candidate_program: RPCModule,
    evaluator: Evaluator,
    rng=None,
) -> RPCRolloutResponse:
    """Evaluate a candidate program on the trainset, using the specified batch size."""
    # Evaluate on the full trainset
    if batch_size >= len(trainset):
        return evaluator.evaluate(
            candidate_program,
            devset=trainset,
        )
    # Or evaluate on a minibatch
    return evaluator.evaluate(
        candidate_program,
        devset=create_minibatch(trainset, batch_size, rng),
    )


def create_n_fewshot_demo_sets(
    student: RPCModule,
    num_candidate_sets: int,
    trainset: list[dict[str, Any]],
    max_labeled_demos: int,
    max_bootstrapped_demos: int,
    evaluator: Evaluator,
    metric_threshold: float | None = None,
    teacher_lm: RPCLM | None = None,
    max_rounds: int = 1,
    labeled_sample: bool = True,
    min_num_samples: int = 1,
    teacher: RPCModule | None = None,
    valset: list[dict[str, Any]] | None = None,
    include_non_bootstrapped: bool = True,
    seed: int = 0,
    rng: random.Random | None = None,
) -> dict[int, list[list[dict[str, Any]]]]:
    """Create N different few-shot demo sets for each predictor in the student module.

    This creates multiple candidate demo sets using various strategies:
    - Zero-shot (no demos)
    - Labels only (just labeled examples)
    - Unshuffled bootstrapped few-shot
    - Multiple shuffled bootstrapped few-shot variants with random sizes

    This is useful for optimizers like RandomSearch that need to evaluate multiple
    demo configurations to find the best one.

    Args:
        student: The student program to create demo sets for.
        num_candidate_sets: Number of candidate sets to create (3 additional sets are added).
        trainset: Training examples to use for creating demos.
        max_labeled_demos: Maximum number of labeled demos per predictor.
        max_bootstrapped_demos: Maximum number of bootstrapped demos per predictor.
        evaluator: Evaluator instance to use.
        metric_threshold: Threshold for accepting bootstrapped demos.
        teacher_lm: Optional LM config for teacher during bootstrapping.
        max_rounds: Number of rounds to attempt bootstrapping per example.
        labeled_sample: Whether to sample labeled demos randomly (vs. taking first K).
        min_num_samples: Minimum number of samples for randomized bootstrap sizes.
        teacher: Optional teacher program for bootstrapping.
        valset: Optional validation set.
        include_non_bootstrapped: Whether to include zero-shot and labels-only sets.
        seed: Random seed for reproducibility.
        rng: Optional random number generator (created from seed if not provided).

    Returns:
        Dictionary mapping predictor index to list of demo lists.
        Each predictor gets a list of candidate demo sets.

    """
    from optimizers.bootstrap import BootstrapFewShot
    from optimizers.vanilla import LabeledFewShot

    demo_candidates = {}

    # Account for the 3 special candidate sets (zero-shot, labels-only, unshuffled)
    num_candidate_sets -= 3

    # Initialize demo_candidates dictionary for each predictor
    predictor_names = list(student.named_predictors.keys())
    for i in range(len(predictor_names)):
        demo_candidates[i] = []

    rng = rng or random.Random(seed)

    # Create each candidate set
    for set_idx in range(-3, num_candidate_sets):
        logger.info(f"Creating demo set {set_idx + 4}/{num_candidate_sets + 3}")

        trainset_copy = list(trainset)

        if set_idx == -3 and include_non_bootstrapped:
            # Candidate 1: Zero-shot (no demos)
            program2 = student.reset_copy()

        elif set_idx == -2 and max_labeled_demos > 0 and include_non_bootstrapped:
            # Candidate 2: Labels only (no bootstrapping)
            # Note: compile() will handle reset_copy internally
            teleprompter = LabeledFewShot(k=max_labeled_demos, evaluator=evaluator)
            program2 = teleprompter.compile(
                student,
                trainset=trainset_copy,
                valset=valset,
                sample=labeled_sample,
            )

        elif set_idx == -1:
            # Candidate 3: Unshuffled bootstrapped few-shot
            # Note: compile() will handle reset_copy internally
            teleprompter = BootstrapFewShot(
                metric_threshold=metric_threshold,
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos,
                max_rounds=max_rounds,
                teacher_lm=teacher_lm,
                evaluator=evaluator,
            )
            program2 = teleprompter.compile(
                student,
                teacher=teacher,
                trainset=trainset_copy,
                valset=valset,
            )

        else:
            # Candidates 4+: Shuffled bootstrapped few-shot with random sizes
            # Note: compile() will handle reset_copy internally
            rng.shuffle(trainset_copy)
            size = rng.randint(min_num_samples, max_bootstrapped_demos)

            teleprompter = BootstrapFewShot(
                metric_threshold=metric_threshold,
                max_bootstrapped_demos=size,
                max_labeled_demos=max_labeled_demos,
                max_rounds=max_rounds,
                teacher_lm=teacher_lm,
                evaluator=evaluator,
            )

            program2 = teleprompter.compile(
                student,
                teacher=teacher,
                trainset=trainset_copy,
                valset=valset,
            )

        # Collect demos from each predictor in the compiled program
        for i, predictor_name in enumerate(predictor_names):
            predictor = program2.named_predictors[predictor_name]
            demo_candidates[i].append(predictor.demos)

    return demo_candidates
