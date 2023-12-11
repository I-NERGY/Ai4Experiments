FROM python:3.9-slim



#RUN apt-get update -y
#RUN apt-get install -y python3-pip python3-dev
RUN pip3 install --upgrade pip

COPY requirements.txt ./requirements.txt
COPY pytorch_predict_ai4eu.py ./pytorch_predict_ai4eu.py
COPY models ./models
COPY checkpoint_200 ./checkpoint_200
COPY prediction_server.py ./
COPY model.proto model_pb2.py model_pb2_grpc.py ./


WORKDIR /

RUN pip3 install -r requirements.txt

EXPOSE 8061 8062

ENTRYPOINT [ "python3","prediction_server.py" ]
