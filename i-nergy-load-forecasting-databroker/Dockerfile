FROM ubuntu:18.04



RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev
RUN pip3 install --upgrade pip
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install tensorflow==2.4.1

RUN pip3 install keras

COPY requirements.txt ./requirements.txt
COPY static ./static
COPY templates ./templates
COPY app.py load_prediction_databroker_server.py ./
COPY model.proto model_pb2.py model_pb2_grpc.py ./


WORKDIR /

RUN pip3 install -r requirements.txt

EXPOSE 8061 8062

ENTRYPOINT [ "python3","load_prediction_databroker_server.py" ]
