import grpc
import yaml
import databroker_pb2_grpc
from google.protobuf import empty_pb2

def run():
    with grpc.insecure_channel(('{}:{}').format(config['client']['host'], config['client']['port'])) as channel:
        stub = databroker_pb2_grpc.GetDailyLoadStub(channel)
        try:
            request = empty_pb2.Empty()
            response = stub.get_daily_load(request)
        except grpc.RpcError as e:
            print(f'grpc error occured:{e}')
        except Exception as e:
            print(f'error occured: {e}')
        else:
            print(f'response is:{response}')


if __name__ == '__main__':
    with open("config.yml", "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
    run()
