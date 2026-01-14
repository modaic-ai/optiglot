import random
from typing import Any

from models import RPCModule

from .teleprompt import RPCTeleprompter


class LabeledFewShot(RPCTeleprompter):
    """A Teleprompter that adds labeled examples to a predictor's prompt."""

    student: RPCModule
    trainset: list[dict[str, Any]]
    k: int

    def __init__(self, k: int = 16, evaluator: Any | None = None):
        """Initialize LabeledFewShot.

        Args:
            k: Number of examples to include in each predictor's prompt.
            evaluator: Optional evaluator instance.

        """
        super().__init__(evaluator=evaluator)
        self.k = k

    def compile(
        self,
        student: RPCModule,
        *,
        trainset: list[dict[str, Any]],
        valset: list[dict[str, Any]] | None = None,
        sample: bool = True,
        **kwargs: Any,
    ) -> RPCModule:
        """Compile the student program by adding labeled examples.

        Args:
            student: The student program to compile.
            trainset: The training set to use for examples.
            valset: Optional validation set, included for API parity.
            sample: Whether to sample examples randomly from the trainset.
            **kwargs: Additional keyword arguments.

        """
        self.student = student.reset_copy()
        self.trainset = trainset
        self.valset = valset

        if len(self.trainset) == 0:
            return self.student

        rng = random.Random(0)

        for predictor in self.student.predictors:
            if sample:
                predictor.demos = rng.sample(
                    self.trainset,
                    min(self.k, len(self.trainset)),
                )
            else:
                predictor.demos = self.trainset[: min(self.k, len(self.trainset))]

        return self.student
