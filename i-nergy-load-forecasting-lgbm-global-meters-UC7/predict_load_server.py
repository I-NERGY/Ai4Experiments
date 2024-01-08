from concurrent import futures
import yaml
import logging
import grpc
import load_prediction_pb2
import load_prediction_pb2_grpc
import os
import predict_total_load as pr

class PredictLoadServicer(load_prediction_pb2_grpc.PredictLoadServicer):
    def GetLoadPrediction(self, request, context):
        logging.info(f'GetLoadPrediction service called with request {request}')
        # hold input in required format by model
        #parse opws thes
        model_input = {} 

        try:
            model_input["n"]=request.hours_ahead
            model_input["ts_id_pred"]=request.ts_id_pred
            model_input["series_uri"]=request.series_uri
        except Exception as e:
            logging.error('error occured while accessing request',e)
            context.set_details('please verify that input is valid')
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return load_prediction_pb2.Prediction()

        logging.info(f'model input is: {model_input}')
        response = pr.MLflowDartsModelPredict("pyfunc_model", model_input)
        logging.info(f'response from model is {response}')
        return load_prediction_pb2.Prediction(
            datetime=response.index.values,
            load=response["Value"],
        )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    load_prediction_pb2_grpc.add_PredictLoadServicer_to_server(PredictLoadServicer(), server)
    server.add_insecure_port(('{}:{}').format(config['server']['host'], config['server']['port']))
    server.start()
    logging.info('load_predict server starts listening at {}:{}'.format(config['server']['host'], config['server']['port']))
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    with open(os.path.join(os.path.dirname(__file__), 'config.yml'), "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
    serve()
