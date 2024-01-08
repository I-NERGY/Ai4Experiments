import grpc
import yaml
import logging
import load_prediction_pb2
import load_prediction_pb2_grpc
import numpy as np
import pandas as pd

# comment following line when providing real data ("news")
#SAMPLE_DATA['news'] = [x if x > 0 else -x for x in SAMPLE_DATA["news"]]

def get_load_prediction(stub):
    
    return stub.GetLoadPrediction(
        load_prediction_pb2.Input(
            hours_ahead="24",
            ts_id_pred="BB6062",
            series_uri="meters/BB6062.csv"
        )
    )


def run():
    with grpc.insecure_channel(
        ("{}:{}").format(config["client"]["host"], config["client"]["port"])
    ) as channel:
        stub = load_prediction_pb2_grpc.PredictLoadStub(channel)
        try:
            response = get_load_prediction(stub)
        except grpc.RpcError as e:
            print(f"grpc error occured:{e.details()} , {e.code().name}")
        except Exception as e:
            print(f"error occured: {e}")
        else:
            df = pd.DataFrame(
                {"Datetime": pd.to_datetime((list(response.datetime))) , "Forecasted Load": list(response.load) },
                # index=pd.to_datetime((list(response.datetime))),
            )
            # df.index.name = "Datetime"
            # df['Time'] = pd.to_datetime(df['datetime']).dt.time
            # df = df.drop('datetime',1)
            print(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    with open("config.yml", "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
    run()
