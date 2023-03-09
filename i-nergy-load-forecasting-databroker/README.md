

**I-NERGY Load Forecasting Databroker**

This is a databroker service used for time series forecasting models. This service is implemented in context of the [I-NERGY project](https://www.i-nergy.eu/). A user interface is provided where users can upload their timeseries as csv files.

**Download & Deployment**

· Use Deploy to local option.

· Extract the downloaded solution.

· Create a Kubernetes namespace. (Kubernetes Cluster shall be available) kubectl create ns

· Inside solution folder locate and run set up script. python .\kubernetes-client-script.py -n

· Verify the status of the created pods. kubectl get pods –n


**Usage**

In case this service is not combined with another service and therefore it is not executed by an orchestrator as a pipeline solution, a gRPC client shall be implemented. If used as a pipeline, it is recommended to utilize our LightGBM Load Forecasting service, also available on this platform. To facilitate this process a
simple example client implementation is provided (see load\_prediction\_databroker\_client.py in Documents section) along with relevant guidelines. 

Users may provide a timeseries in the form of a .csv file (for example series.csv), that has two columns, one of which must be named Load and the other should be named Date. This could be done in the UI of this service, exposed at 8062 port.

In order to execute the client:

· Install required dependencies (I.e., pandas,numpy,grcpio,logging)

· Generate the imported classes. (Classes are also available inside container and therefore can be copied, if users have access to container.)

	install grpcio-tools.

	locate file load\_prediction.proto inside folder microservice

	create classes: python3 -m grpc\_tools.protoc -I. --python\_out=. --
	grpc\_python\_out=. load\_prediction.proto

· Configure request's payload as described above. • replace values of host & port to the ones of the deployed service.

· Run client: python3 client.py





