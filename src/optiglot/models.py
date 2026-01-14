from collections.abc import Callable
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

FieldType = Literal["string", "number", "integer", "boolean", "array", "object"] | dict


class RPCField(BaseModel):
    prefix: str
    type: FieldType
    kind: Literal["input", "output"]
    description: str | None = None


class RPCSignature(BaseModel):
    instructions: str
    input_fields: dict[str, RPCField]
    output_fields: dict[str, RPCField]

    def with_instructions(self, instructions: str) -> "RPCSignature":
        return self.model_copy(update={"instructions": instructions})


class RPCDemo(BaseModel):
    input: dict[str, Any]
    output: dict[str, Any]


class RPCLM(BaseModel):
    model: str
    model_type: Literal["chat", "text", "responses"] = "chat"
    temperature: float | None = None
    rollout_id: int | None = None


def _deepcopy(self: "RPCPredict" | "RPCModule") -> "RPCPredict" | "RPCModule":
    return self.model_copy(deep=True)


class RPCPredict(BaseModel):
    signature: RPCSignature
    demos: list[dict[str, Any]] = []
    lm: RPCLM | None = None

    def deepcopy(self) -> "RPCPredict":
        return _deepcopy(self)

    def reset(self):
        self.lm = None
        self.demos = []


class RPCModule(BaseModel):
    """A module is a collection of predictors that are executed together."""

    named_predictors: dict[str, RPCPredict]

    @property
    def predictors(self) -> list[RPCPredict]:
        return list(self.named_predictors.values())

    def deepcopy(self) -> "RPCModule":
        return _deepcopy(self)

    def reset_copy(self) -> "RPCModule":
        new_instance = self.deepcopy()

        for param in new_instance.predictors:
            param.reset()

        return new_instance


class Prediction(BaseModel):
    """A prediction is the result of a single predictor's execution."""

    model_config = ConfigDict(populate_by_name=True)
    output: Any | None
    error_type: str | None = Field(default=None, alias="errorType")
    error_message: str | None = Field(default=None, alias="errorMessage")
    error_traceback: str | None = Field(default=None, alias="errorTraceback")


class RPCRolloutRequest(BaseModel):
    """Request to roll out a program. or canidate programs.

    Attributes:
        canidates: The canidate programs to roll out. (Usually just one.)
        examples: The examples to roll out the programs on.
        trace: Whether to trace the rollout.
        lm: The language model to use for the rollout if you would like to change the model settings during optimization.

    """

    canidates: list[RPCModule]
    examples: list[dict[str, Any]]
    trace: bool = False
    lm: RPCLM | None = None


class RPCTraceStep(BaseModel):
    """Represents a single predictor's execution within a program's forward pass."""

    predictor_name: str
    inputs: dict[str, Any]
    prediction: Prediction


class RPCUsage(BaseModel):
    """Tracks token consumption for the rollout."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class RPCRolloutResult(BaseModel):
    """The result of running ONE candidate on ONE example."""

    example: dict[str, Any]
    prediction: Prediction
    # The numeric value from the metric function run on the server
    score: float | None = None
    feedback: str | None = None
    # The sequence of predictor calls (essential for Bootstrap/MIPRO/etc.)
    trace: list[RPCTraceStep] | None = None
    usage: RPCUsage | None = None


class RPCRolloutResponse(BaseModel):
    results: list[RPCRolloutResult]

    @property
    def total_usage(self) -> RPCUsage | None:
        if not self.results:
            return None
        prompt_tokens = sum(
            result.usage.prompt_tokens for result in self.results if result.usage
        )
        completion_tokens = sum(
            result.usage.completion_tokens for result in self.results if result.usage
        )
        total_tokens = sum(
            result.usage.total_tokens for result in self.results if result.usage
        )
        return RPCUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )


RPCMetric = Callable[[dict[str, Any], Prediction], float]
