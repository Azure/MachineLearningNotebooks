import os

import numpy as np
from azureml.monitoring import ModelDataCollector
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.schema_decorators import input_schema, output_schema
# sklearn.externals.joblib is removed in 0.23
from sklearn import __version__ as sklearnver
from packaging.version import Version
if Version(sklearnver) < Version("0.23.0"):
    from sklearn.externals import joblib
else:
    import joblib


def init():
    global model
    global inputs_dc
    inputs_dc = ModelDataCollector('elevation-regression-model.pkl', designation='inputs',
                                   feature_names=['latitude', 'longitude', 'temperature', 'windAngle', 'windSpeed'])
    # note here "elevation-regression-model.pkl" is the name of the model registered under
    # this is a different behavior than before when the code is run locally, even though the code is the same.
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'elevation-regression-model.pkl')
    model = joblib.load(model_path)


input_sample = np.array([[30, -85, 21, 150, 6]])
output_sample = np.array([8.995])


@input_schema('data', NumpyParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        inputs_dc.collect(data)
        result = model.predict(data)
        # you can return any datatype as long as it is JSON-serializable
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
