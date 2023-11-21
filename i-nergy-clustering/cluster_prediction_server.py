from concurrent import futures
import yaml
import logging
import grpc
import clustering_pb2
import clustering_pb2_grpc
import assign_cluster as pr


class PredictClusterServicer(clustering_pb2_grpc.PredictClusterServicer):
    def GetClusterPrediction(self, request, context):
        logging.info(f'GetClusterPrediction service called with request {request}')

        try:
            load_values_24h = request.load_values
            date = request.date
        except Exception as e:
            logging.error('error occurred while accessing request', e)
            context.set_details('please verify that input is valid')
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return clustering_pb2.Prediction()
        else:
            try:
                cluster_found = pr.assign_cluster(load_values_24h)
                logging.info(f'response from model is {cluster_found}')
            except FileNotFoundError as fnf:
                logging.error('Artifact not found', fnf)
                context.set_code(grpc.StatusCode.INTERNAL)
                return clustering_pb2.Prediction()
            except Exception as e:
                logging.error('Error occurred while calculating cluster', e)
                context.set_code(grpc.StatusCode.INTERNAL)
                return clustering_pb2.Prediction()
            else:
                return clustering_pb2.Prediction(date=date, cluster=cluster_found)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    clustering_pb2_grpc.add_PredictClusterServicer_to_server(PredictClusterServicer(), server)
    server.add_insecure_port(('{}:{}').format(config['server']['host'], config['server']['port']))
    server.start()
    logging.info(
        'Clustering prediction server listening at {}:{}'.format(config['server']['host'], config['server']['port']))
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    with open("config.yml", "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
    serve()
