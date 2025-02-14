from abc import ABC as AbstractBaseClass, abstractmethod
from typing import Callable

from fundar_llms.api.interface import PlainPromptInterface, BaseResponse, LlmArgs

class Llm(PlainPromptInterface):
    pass

class Metric(AbstractBaseClass):
    evaluate: Callable

class LlmMetric(Metric):
    @abstractmethod
    def __init__(self, *args, llm: Llm, **kwargs):
        pass

# ==============================================================================

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, TypedDict

class TokenUsage(TypedDict):
    total: int
    prompt: int
    completion: int
    numRequests: int
    cached: int

@dataclass
class GradingResult:
    score: float
    reason: str
    
    passed            : bool                            = field(default=False)
    named_scores      : Optional[Dict[str, float]]      = field(default=None)
    tokens_used       : Optional['TokenUsage']          = field(default=None)
    component_results : Optional[List['GradingResult']] = field(default=None)
    assertion         : Optional[Any]                   = field(default=None)
    comment           : Optional[str]                   = field(default=None)
    suggestions       : Optional[List[Any]]             = field(default=None)
    metadata          : Optional[Dict[str, Any]]        = field(default=None)

    @classmethod
    def from_response(cls, response: str) -> 'GradingResult':
        # TODO: Definir un formato que sea facilmente parseable por esta función
        # y simultáneamente que sea fácil de generar para el LLM como formato
        # de salida.
        raise NotImplementedError
    
class LlmRubric(LlmMetric):
    def __init__(self, llm: Llm, rubric: str, llm_args: LlmArgs):
        self.llm = llm

        # TODO: Acá probablemente haya que agregar texto para forzar al llm
        # que genere un formato valido de salida (ej. JSON) para que finalmente
        # pueda ser parseado como un GradingResult válido.
        self.args = llm_args | dict(prompt=rubric)
    
    def evaluate(self, x: str) -> GradingResult:
        response = self.llm.generate(**self.args)
        return GradingResult(response)
    
class RougeN(Metric):
    def __init__(self, threshold: float):
        self.threshold = threshold

    def evaluate(self, x: str) -> GradingResult:
        ...