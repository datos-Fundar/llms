"""
This type stub file was generated by pyright.
"""

from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from collections.abc import Iterable
from typing import Optional, TypeVar
from fundar_llms.utils.common import allow_opaque_constructor

T = TypeVar('T')
DEFAULT_PDF_LOADER = PyPDFLoader
def load_document(filepath: str, loader=..., seed_id: Optional[int] = ...) -> list[Document]:
    ...

DEFAULT_RCT_SPLITTER = ...
@allow_opaque_constructor(splitter=RecursiveCharacterTextSplitter)
def split_document(xs, splitter=..., seed_id: Optional[int] = ...) -> list[Document]:
    ...

def load_and_split(filepath: str, loader=..., splitter=..., seed_id: Optional[int] = ..., flatten=...): # -> list[Document] | list[list[Document]]:
    ...

_sentence_transformer_obj = ...
def SentenceTransformer(*args, **kwargs): # -> SentenceTransformer:
    """
    Default args:
        - model: sentence-transformers/all-mpnet-base-v2
        - device: auto (cuda if available)
    """
    ...

def encode_with_multiprocessing(transformer, pool): # -> Callable[..., NDArray[Any]]:
    ...

@allow_opaque_constructor(sentence_transformer=SentenceTransformer)
def vectorize_document(x: str | Document | Iterable[Document | str], sentence_transformer=..., uid=..., additional_metadata=..., devices=...): # -> list[dict[str, List[Tensor] | ndarray[Any, Any] | Tensor | NDArray[Any] | Any]] | list[dict[str, Document | str | dict[Any, Any] | UUID | Tensor]]:
    ...

