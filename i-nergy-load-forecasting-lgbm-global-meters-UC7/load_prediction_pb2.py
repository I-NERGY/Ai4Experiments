# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: load_prediction.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x15load_prediction.proto\"D\n\x05Input\x12\x12\n\nseries_uri\x18\x01 \x01(\t\x12\x13\n\x0bhours_ahead\x18\x02 \x01(\t\x12\x12\n\nts_id_pred\x18\x03 \x01(\t\",\n\nPrediction\x12\x10\n\x08\x64\x61tetime\x18\x01 \x03(\x01\x12\x0c\n\x04load\x18\x02 \x03(\x01\x32\x37\n\x0bPredictLoad\x12(\n\x11GetLoadPrediction\x12\x06.Input\x1a\x0b.Predictionb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'load_prediction_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_INPUT']._serialized_start=25
  _globals['_INPUT']._serialized_end=93
  _globals['_PREDICTION']._serialized_start=95
  _globals['_PREDICTION']._serialized_end=139
  _globals['_PREDICTLOAD']._serialized_start=141
  _globals['_PREDICTLOAD']._serialized_end=196
# @@protoc_insertion_point(module_scope)