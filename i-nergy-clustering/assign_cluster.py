from typing import List

import mlflow.sklearn
import numpy as np
from pickle import load


def assign_cluster(load_values: List[float]) -> int:
    scaler = load(open('scaler.pkl', 'rb'))
    arr_1d = np.array(load_values)
    arr_2d = np.reshape(arr_1d, (1, -1))
    model_input = scaler.transform(arr_2d)
    kmeans = mlflow.sklearn.load_model(model_uri='kmeansEuclidean14')
    cluster = kmeans.predict(model_input)
    return cluster[0]
