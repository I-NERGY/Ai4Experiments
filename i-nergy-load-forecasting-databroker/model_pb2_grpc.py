# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import model_pb2 as model__pb2


class DatabrokerStub(object):
    """Define the service
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.databroker = channel.unary_unary(
                '/Databroker/databroker',
                request_serializer=model__pb2.Empty.SerializeToString,
                response_deserializer=model__pb2.Features.FromString,
                )


class DatabrokerServicer(object):
    """Define the service
    """

    def databroker(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_DatabrokerServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'databroker': grpc.unary_unary_rpc_method_handler(
                    servicer.databroker,
                    request_deserializer=model__pb2.Empty.FromString,
                    response_serializer=model__pb2.Features.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'Databroker', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Databroker(object):
    """Define the service
    """

    @staticmethod
    def databroker(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Databroker/databroker',
            model__pb2.Empty.SerializeToString,
            model__pb2.Features.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
