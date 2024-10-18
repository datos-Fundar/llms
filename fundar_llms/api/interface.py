from typing import Protocol, Optional, Any
from dataclasses import dataclass

# TODO: Mover esto a otro modulo de tipos que esté más arriba.
class Base64(str): ...
class Context(Any): ...

response_dataclass = dataclass(
    init            = True,
    repr            = True,
    eq              = True,
    order           = False,
    unsafe_hash     = True,
    frozen          = True,
    match_args      = False,
    kw_only         = True,
    slots           = True,
    weakref_slot    = False,
)

@response_dataclass
class BaseResponse:
    model: str
    prompt: str
    system: str
    response: str
    total_duration: int
    load_duration: Optional[int]
    done: Optional[bool]
    done_reason: Optional[str]
    context: Optional[Context]


class PlainPromptInterface(Protocol):
    def generate(
            self,
            model: str,
            prompt: str,
            raw: Optional[bool] = None,
            image: Optional[Base64] = None,
            suffix: Optional[str] = None,
            format: Optional[str] = None,
            system: Optional[str] = None,
            context: Optional[Context] = None,
            stream: Optional[bool] = None,
            num_ctx: Optional[int] = None,
            num_predict: Optional[int] = None,
            temperature: Optional[float] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            *args,
            **kwargs
    ) -> Any: ...


