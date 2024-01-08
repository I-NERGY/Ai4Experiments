# [I-NERGY](https://www.i-nergy.eu/) Energy Efficiency Investments Service

I-NERGY energy efficiency investments service is based on a physics-informed model for predicting the electricity consumption of a building based on the heat losses of the envelope.


## Download and installation

* Use *Deploy to local* option.
* Extract the downloaded solution.
* Create a Kubernetes namespace. (Kubernetes Cluster shall be available)  
`
kubectl create ns <namespace_name>
`
* Inside solution folder locate and run set up script.  
`
python .\kubernetes-client-script.py -n <namespace_name>
`
* Verify the status of the created pods.  
`
kubectl get pods –n <namespace_name>
`

## Usage

In case this service is not combined with another service and therefore it is not executed by an orchestrator as a pipeline solution, a gRPC client shall be implemented. To facilitate this process a simple example client's implementation is provided (see client.py in Documents section) along with relevant guidelines.

```python
import grpc
import logging
import model_pb2
import model_pb2_grpc
import numpy as np
import pandas as pd
import json


def get_EnergyConsumption(stub):
    params = {
"useful_area": 13077,
"floors": 9,
"apartments": 252,
"total_area": 15673,
"serie": 101,
"Heavy": 1,
"Light": 0
}

    return stub.EnergyConsumption(
        model_pb2.Input(
            input_message = json.dumps(params)
        )
    )


def run():
    with grpc.insecure_channel(port_address) as channel:

        stub = model_pb2_grpc.PredictStub(channel)
        try:
            response = get_EnergyConsumption(stub)
        except grpc.RpcError as e:
            print(f"grpc error occurred: {e.details()}, {e.code().name}")
        except Exception as e:
            print(f"error occurred: {e}")
        else:
            output = json.loads(response.output_message)
            print(output)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    run()
```

Users are able to configure the following fields of the request's payload:

* *useful_area* : provide a positive value corresponding to the useful area of the building.
* *floors* : provide value equal or greater than 1, referring to the number of floors of the building.
* *apartments* : provide value equal or greater than 1, referring to the number of apartments of the building.
* *total_area* : provide a positive value corresponding to the total area of the building.
* *serie* : replace this value with the serie number of the building.
* *Heavy* : provide o boolean number (1 or 0) to specify if the building is constructed with heavy materials. When heavy is 1 , light must be 0.
* *Light* : provide o boolean number (1 or 0) to specify if the building is constructed with light materials. When light is 1 , heavy must be 0.

In order to locally execute client:

* install required dependencies (i.e., `grpcio`, `pandas`, `pyyaml`).
* generate the imported classes. (classes are also available inside container and therefore can be copied, if users have access to container.)
  * install `grpcio-tools`.
  * locate file load_prediction.proto inside folder *microservice*
  * create classes: `python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. load_prediction.proto`
* configure request's payload as described above.
* replace values of host & port to the ones of the deployed service.
* run client: `python3 client.py`
