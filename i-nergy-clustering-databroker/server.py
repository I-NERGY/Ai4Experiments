# implementation of grpc route prediction server
from concurrent import futures
import pandas as pd
import yaml
import logging
import grpc
import databroker_pb2
import databroker_pb2_grpc
import csv_reader

file_reader = None


class GetDailyLoadServicer(databroker_pb2_grpc.GetDailyLoadServicer):
    def get_daily_load(self, request, context):
        logging.info(f'get_route service called with request: {str(request)} , type of: {type(request)}')
        response = databroker_pb2.Input()
        if (file_reader):
            if (file_reader.total == file_reader.current):
                logging.info('File end reached')
                file_reader.reset()
                context.set_details("No more data")
                context.set_code(grpc.StatusCode.OUT_OF_RANGE)
                return response
            else:
                logging.info('Route exists')
                daily_loads = file_reader.get_fixed_slice()
                logging.info(f'route is: {daily_loads}')
                # daily_loads single element at this service
                for day in daily_loads:
                    print('Day', day)
                    try:
                        response = databroker_pb2.Input(
                            date=day[0],
                            load_values=day[1:]
                        )
                    except Exception as e:
                        logging.error('Error occured', e)
                        context.set_details('Please check csv file')
                        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                print(response)
                return response
        else:
            context.set_details("File does not exist")
            context.set_code(grpc.StatusCode.NOT_FOUND)
            return response


def serve():
    logging.info('csv_server starts listening at {}:{}'.format(config['server']['host'], config['server']['port']))
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    databroker_pb2_grpc.add_GetDailyLoadServicer_to_server(GetDailyLoadServicer(), server)
    server.add_insecure_port(('{}:{}').format(config['server']['host'], config['server']['port']))
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    with open("config.yml", "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
    file_reader = csv_reader.Reader()
    serve()
