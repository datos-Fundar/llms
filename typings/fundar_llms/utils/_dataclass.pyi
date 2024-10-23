"""
This type stub file was generated by pyright.
"""

from abc import ABC as AbstractBaseClass
from typing import Any

class DataclassDictUtilsMixin(AbstractBaseClass):
    @classmethod
    def from_dict(cls, data: dict[str, Any]): # -> Self:
        ...
    
    def to_dict(self, exclude=..., compact=...) -> dict[str, Any]:
        ...
    


