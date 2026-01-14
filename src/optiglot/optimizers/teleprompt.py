from typing import Any

import requests
from models import RPCModule
from settings import settings
from utils import Evaluator


class RPCTeleprompter:
    def __init__(self, evaluator: Evaluator | None = None, **kwargs: Any):
        self.evaluator = evaluator or Evaluator()
        self.kwargs = kwargs

    def compile(
        self,
        student: RPCModule,
        *,
        trainset: list[dict[str, Any]],
        valset: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> RPCModule:
        """Optimize the student program.

        Args:
            student: The student program to optimize.
            trainset: The training set to use for optimization.
            valset: The validation set to use for optimization.

        Returns:
            The optimized student program.

        """
        raise NotImplementedError

    def compile_and_send(
        self,
        student: RPCModule,
        *,
        trainset: list[dict[str, Any]],
        valset: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> dict:
        """Compile the student program and send final module back to the server.

        Args:
            student: The student program to optimize.
            trainset: The training set to use for optimization.
            valset: The validation set to use for optimization.
            endpoint: The server endpoint to send the module to.

        Returns:
            The optimized student program.

        """
        compiled = self.compile(
            student,
            trainset=trainset,
            valset=valset,
            **kwargs,
        )

        response = requests.post(
            f"{settings.host_url}/send",
            json=compiled.model_dump(),
        )
        response.raise_for_status()
        return response.json()

    def compile_dspy(
        self,
        student: RPCModule,
        *,
        trainset: list[dict[str, Any]],
        valset: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> RPCModule:
        """Compile a DSPy program"""
        raise NotImplementedError

    def get_params(self) -> dict[str, Any]:
        """Get the parameters of the teleprompter.

        Returns:
            The parameters of the teleprompter.

        """
        return self.__dict__
