
from concurrent import futures
import time
import logging
# import the generated classes :
import model_pb2
import model_pb2_grpc
import grpc
import pandas as pd
import os
# import the function we made :
import inference as inf

port = 8061

# create a class to define the server functions, derived from

class PredictLoadServicer(model_pb2_grpc.PredictLoadServicer):
    def  MLflowDartsModelPredict(self, request, context):
        logging.info(f'MLflowDartsModelPredict service called with request { request }')
        model_input = {}
        df = pd.DataFrame(
            {"Date": pd.to_datetime((list(request.Datetime))), "Load": list(request.Load) }
        )
        #df['Date'] = pd.to_datetime(df['datetime']).dt.time

        #df = df.drop('datetime', 1)
        df.to_csv('series.csv', index=False)
        try:
            #model_input["pyfunc_model_folder"]=request.pyfunc_model_folder
            #model_input["forecast_horizon"]=request.forecast_horizon
            model_input["series_uri"]="./series.csv"
            #model_input["future_covariates_uri"]=request.future_covariates_uri
            #model_input["past_covariates_uri"]=request.past_covariates_uri
            #model_input["roll_size"]=request.roll_size
            #model_input["batch_size"]=request.batch_size
        except Exception as e:
            logging.error('error occured while accessing request',e)
            context.set_details('please verify that the input is valid')
            context.set_code(grpc.StatusCodes.INVALID_ARGUMENT)
            return model_pb2.Prediction()

        logging.info(f'model input is: {model_input}')

        response = inf.MLflowDartsModelPredict(model_input["series_uri"])
        predict = model_pb2.Prediction()
        flat_list = [item for sublist in response.values.tolist() for item in sublist]


        predict.Load.extend(flat_list)

        predict.Datetime.extend(response.index.values)

        logging.info(f'response from model is {response}')

        return predict

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_pb2_grpc.add_PredictLoadServicer_to_server(PredictLoadServicer(), server)
    server.add_insecure_port("[::]:{}".format(port))
    server.start()
    logging.info("MLflowDartsModelPredict server starts listening at " + str(port))
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    #with open("config.yaml", "r") as ymlfile:
    #    config = yaml.safe_load(ymlfile)
    serve()
