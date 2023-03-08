import grpc
from concurrent import futures
import threading

# import the generated classes :
import model_pb2
import model_pb2_grpc
from app import app_run, get_parameters
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
port = 8061


# create a class to define the server functions, derived from
class DatabrokerServicer(model_pb2_grpc.DatabrokerServicer):

    def databroker(self, request, context):
        parameters = get_parameters()
        #logger.debug(parameters)
        #parameters.astype("float64")
        logger.debug("Connecting to databroker")
        #flat_list = [item for sublist in parameters.values.tolist() for item in sublist]
        response = model_pb2.Features(Load=parameters["Load"], Datetime=parameters["Date"]
                                      )
        logger.debug(response)
        return response


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_pb2_grpc.add_DatabrokerServicer_to_server(DatabrokerServicer(), server)
    server.add_insecure_port('[::]:{}'.format(port))
    logger.debug("Start server")
    server.start()
    logger.debug("Start databroker UI")
    threading.Thread(target=app_run()).start()
    server.wait_for_termination()
    logger.debug("Threads ended")


if __name__ == '__main__':
    logging.basicConfig()
    serve()
