# 05-upload-data.py
from azureml.core import Workspace
ws = Workspace.from_config()
datastore = ws.get_default_datastore()
datastore.upload(src_dir='./data',
                 target_path='datasets/cifar10',
                 overwrite=True)
