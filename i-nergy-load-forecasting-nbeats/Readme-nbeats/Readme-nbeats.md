**I-NERGY Load Forecasting Inference Server – NBEATS Model** 

This is a time series forecasting service for predicting of the Portuguese aggregated electricity load series (15-min resolution, 24hr forecasting horizon). This service is based on an NBEATS model trained in the context of[ I-NERGY ](https://www.i-nergy.eu/)project. The model has been trained on the Portuguese timeseries from 2013 to 2019 validated on year 2020 and tested on 2021 with Mean Absolute Percentage Error (MAPE) = 2.35%. No time covariates or external variables have been included in the model. The lookback window of the model is 10 days. The model can be used to produce forecasts for periods from 2022 and later for Portugal. Other transmission system operators may use it as well, however expecting lower performance in general. No external variables have been considered. Please keep in mind that the effects of the pandemic on national loads can negatively affect the model’s performance.

**Download & Deployment** 

- Use Deploy to local option.
- Extract the downloaded solution. 
- There is a possibility to execute the model with different forecast horizon, roll size and batch size. Forecast horizon should be less than 2880, roll size lower or equal to the forecast horizon and batch size can reach up to 64. All three arguments should be integer numbers. ?  
- In order to do so the user must enter the aforementioned values as environment variables in the deployment.yaml downloaded by the AI4EU Experiments platform. An example with the default values is shown below.



- Create a Kubernetes namespace. (Kubernetes Cluster shall be available) **kubectl create ns**
- Inside solution folder locate and run set up script. **python .\kubernetes-client-script.py -n**
- Verify the status of the created pods. **kubectl get pods –n**


**Usage** 

In case this service is not combined with another service and therefore it is not executed by an orchestrator as a pipeline solution, a gRPC client shall be implemented. If used as a pipeline, it is recommended to utilize our i-nergy-load-forecasting-databroker service, which is also available on this platform. To facilitate this process a simple example client implementation is provided (see load\_prediction\_client.py in Documents section) along with relevant guidelines. 

Users may provide the path to a timeseries in the form of a .csv file (for example series.csv), that has two columns, one of which must be named Load and the other should be named Date. 

In order to execute the client: 

- Install required dependencies (i.e., pandas, numpy, grcpio, logging) 
- Generate the imported classes. (Classes are also available inside container and therefore can be copied, if users have access to container.)  

	install grpcio-tools.  

	locate file model.proto inside folder microservice 

	create classes: python3 -m grpc\_tools.protoc -I. --python\_out=. --  grpc\	_python\_out=. model.proto 

- Configure request's payload as described above.  
- Replace values of host & port to the ones of the deployed service.  
- Run client: python3 load\_prediction\_client.py 
