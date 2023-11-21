from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Input(_message.Message):
    __slots__ = ["date", "load_values"]
    DATE_FIELD_NUMBER: _ClassVar[int]
    LOAD_VALUES_FIELD_NUMBER: _ClassVar[int]
    date: str
    load_values: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, date: _Optional[str] = ..., load_values: _Optional[_Iterable[float]] = ...) -> None: ...

class Prediction(_message.Message):
    __slots__ = ["date", "cluster"]
    DATE_FIELD_NUMBER: _ClassVar[int]
    CLUSTER_FIELD_NUMBER: _ClassVar[int]
    date: str
    cluster: int
    def __init__(self, date: _Optional[str] = ..., cluster: _Optional[int] = ...) -> None: ...
