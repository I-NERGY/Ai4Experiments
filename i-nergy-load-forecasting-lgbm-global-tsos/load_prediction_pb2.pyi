from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Input(_message.Message):
    __slots__ = ["series_uri", "hours_ahead", "ts_id_pred"]
    SERIES_URI_FIELD_NUMBER: _ClassVar[int]
    HOURS_AHEAD_FIELD_NUMBER: _ClassVar[int]
    TS_ID_PRED_FIELD_NUMBER: _ClassVar[int]
    series_uri: str
    hours_ahead: str
    ts_id_pred: str
    def __init__(self, series_uri: _Optional[str] = ..., hours_ahead: _Optional[str] = ..., ts_id_pred: _Optional[str] = ...) -> None: ...

class Prediction(_message.Message):
    __slots__ = ["datetime", "load"]
    DATETIME_FIELD_NUMBER: _ClassVar[int]
    LOAD_FIELD_NUMBER: _ClassVar[int]
    datetime: _containers.RepeatedScalarFieldContainer[float]
    load: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, datetime: _Optional[_Iterable[float]] = ..., load: _Optional[_Iterable[float]] = ...) -> None: ...
