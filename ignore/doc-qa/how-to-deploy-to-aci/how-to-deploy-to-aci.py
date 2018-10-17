#!/usr/bin/env python
# coding: utf-8

import azureml.core
print('SDK version' + azureml.core.VERSION)

# PREREQ: load workspace info
# import azureml.core

# <loadWorkspace>
from azureml.core import Workspace
ws = Workspace.from_config()
# </loadWorkspace>

scorepy_content = "import json\nimport numpy as np\nimport os\nimport pickle\nfrom sklearn.externals import joblib\nfrom sklearn.linear_model import LogisticRegression\n\nfrom azureml.core.model import Model\n\ndef init():\n    global model\n    # retreive the path to the model file using the model name\n    model_path = Model.get_model_path('sklearn_mnist')\n    model = joblib.load(model_path)\n\ndef run(raw_data):\n    data = np.array(json.loads(raw_data)['data'])\n    # make prediction\n    y_hat = model.predict(data)\n    return json.dumps(y_hat.tolist())"
print(scorepy_content)
with open("score.py","w") as f:
    f.write(scorepy_content) 


# PREREQ: create environment file
from azureml.core.conda_dependencies import CondaDependencies 

myenv = CondaDependencies()
myenv.add_conda_package("scikit-learn")

with open("myenv.yml","w") as f:
    f.write(myenv.serialize_to_string())

#<configImage>
from azureml.core.image import ContainerImage

image_config = ContainerImage.image_configuration(execution_script = "score.py",
                                                  runtime = "python",
                                                  conda_file = "myenv.yml",
                                                  description = "Image with mnist model",
                                                  tags = {"data": "mnist", "type": "classification"}
                                                 )
#</configImage>

# <configAci>
from azureml.core.webservice import AciWebservice

aciconfig = AciWebservice.deploy_configuration(cpu_cores = 1, 
                                               memory_gb = 1, 
                                               tags = {"data": "mnist", "type": "classification"}, 
                                               description = 'Handwriting recognition')
# </configAci>

#<registerModel>
from azureml.core.model import Model

model_name = "sklearn_mnist"
model = Model.register(model_path = "sklearn_mnist_model.pkl",
                        model_name = model_name,
                        tags = {"data": "mnist", "type": "classification"},
                        description = "Mnist handwriting recognition",
                        workspace = ws)
#</registerModel>

# <retrieveModel>
from azureml.core.model import Model

model_name = "sklearn_mnist"
model=Model(ws, model_name)
# </retrieveModel>


# ## DEPLOY FROM REGISTERED MODEL

# <option2Deploy>
from azureml.core.webservice import Webservice

service_name = 'aci-mnist-2'
service = Webservice.deploy_from_model(deployment_config = aciconfig,
                                       image_config = image_config,
                                       models = [model], # this is the registered model object
                                       name = service_name,
                                       workspace = ws)
service.wait_for_deployment(show_output = True)
print(service.state)
# </option2Deploy>

service.delete()

# ## DEPLOY FROM IMAGE


# <option3CreateImage>
from azureml.core.image import ContainerImage

image = ContainerImage.create(name = "myimage1",
                              models = [model], # this is the registered model object
                              image_config = image_config,
                              workspace = ws)

image.wait_for_creation(show_output = True)
# </option3CreateImage>

# <option3Deploy>
from azureml.core.webservice import Webservice

service_name = 'aci-mnist-13'
service = Webservice.deploy_from_image(deployment_config = aciconfig,
                                            image = image,
                                            name = service_name,
                                            workspace = ws)
service.wait_for_deployment(show_output = True)
print(service.state)
# </option3Deploy>

service.delete()


# ## DEPLOY FROM MODEL FILE
# First change score.py!



scorepy_content = "import json\nimport numpy as np\nimport os\nimport pickle\nfrom sklearn.externals import joblib\nfrom sklearn.linear_model import LogisticRegression\n\nfrom azureml.core.model import Model\n\ndef init():\n    global model\n    # retreive the path to the model file using the model name\n    model_path = Model.get_model_path('sklearn_mnist_model.pkl')\n    model = joblib.load(model_path)\n\ndef run(raw_data):\n    data = np.array(json.loads(raw_data)['data'])\n    # make prediction\n    y_hat = model.predict(data)\n    return json.dumps(y_hat.tolist())"
with open("score.py","w") as f:
    f.write(scorepy_content) 



# <option1Deploy>
from azureml.core.webservice import Webservice

service_name = 'aci-mnist-1'
service = Webservice.deploy(deployment_config = aciconfig,
                                image_config = image_config,
                                model_paths = ['sklearn_mnist_model.pkl'],
                                name = service_name,
                                workspace = ws)

service.wait_for_deployment(show_output = True)
print(service.state)
# </option1Deploy>

# <testService>
# Load Data
import os
import urllib

os.makedirs('./data', exist_ok = True)

urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', filename = './data/test-images.gz')

from utils import load_data
X_test = load_data('./data/test-images.gz', False) / 255.0

from sklearn import datasets
import numpy as np
import json

# find 5 random samples from test set
n = 5
sample_indices = np.random.permutation(X_test.shape[0])[0:n]

test_samples = json.dumps({"data": X_test[sample_indices].tolist()})
test_samples = bytes(test_samples, encoding = 'utf8')

# predict using the deployed model
prediction = service.run(input_data = test_samples)
print(prediction)
# </testService>

# <deleteService>
service.delete()
# </deleteService>




