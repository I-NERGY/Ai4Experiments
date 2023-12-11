import grpc
import yaml
import logging
import clustering_pb2
import clustering_pb2_grpc
import pandas as pd


def get_cluster_prediction(stub):
    sample_df = pd.read_csv('sample.csv')
    sample = sample_df.iloc[0]
    sample_date = sample['date']
    sample_load_24h = sample.drop('date').values
    return stub.GetClusterPrediction(
        clustering_pb2.Input(
            date=sample_date,
            load_values=sample_load_24h
        )
    )


def run():
    with grpc.insecure_channel(
            ("{}:{}").format(config["client"]["host"], config["client"]["port"])
    ) as channel:
        stub = clustering_pb2_grpc.PredictClusterStub(channel)
        try:
            response = get_cluster_prediction(stub)
        except grpc.RpcError as e:
            print(f"grpc error occurred:{e}")
        except Exception as e:
            print(f"error occurred: {e}")
        else:
            print(f"Cluster assignment for {response.date} is {response.cluster}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    with open("config.yml", "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
    run()
