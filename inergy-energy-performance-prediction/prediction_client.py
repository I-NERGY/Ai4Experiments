import grpc
import logging
import model_pb2
import model_pb2_grpc
import numpy as np
import pandas as pd
import json

port_address = 'localhost:8061'


def get_EnergyConsumption(stub):
    params = {
"useful_area": 13077,
"floors": 9,
"apartments": 252,
"total_area": 15673,
"serie": 101,
"Heavy": 1,
"Light": 0
}

    return stub.EnergyConsumption(
        model_pb2.Input(
            input_message = json.dumps(params)
        )
    )


def run():
    with grpc.insecure_channel(port_address) as channel:

        stub = model_pb2_grpc.PredictStub(channel)
        try:
            response = get_EnergyConsumption(stub)
        except grpc.RpcError as e:
            print(f"grpc error occurred: {e.details()}, {e.code().name}")
        except Exception as e:
            print(f"error occurred: {e}")
        else:
            output = json.loads(response.output_message)
            print(output)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    run()
