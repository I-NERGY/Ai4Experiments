import grpc
import logging
import model_pb2
import model_pb2_grpc
import numpy as np
import pandas as pd

port_address = 'localhost:8061'


def get_MLflowDartsModelPrediction(stub):
    df = pd.read_csv("../ENG/series.csv")

    return stub.MLflowDartsModelPredict(
        model_pb2.Input(
            Load = df["Load"],
            Datetime = df["Date"]
        )
    )

def run():
    with grpc.insecure_channel(port_address) as channel:

        stub = model_pb2_grpc.PredictLoadStub(channel)
        try:
            response = get_MLflowDartsModelPrediction(stub)
        except grpc.RpcError as e:
            print(f"grpc error occurred: {e.details()}, {e.code().name}")
        except Exception as e:
            print(f"error occurred: {e}")
        else:
            df = pd.DataFrame(
                {"Datetime": pd.to_datetime((list(response.Datetime))), "Predicted Load": list(response.Load) }
            )
            
            print(df)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    run()
