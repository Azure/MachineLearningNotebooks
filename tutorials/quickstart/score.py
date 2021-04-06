import os
import torch
import json
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def init():
    global net
    global classes

    model_filename = 'cifar_net.pth'
    model_path = os.path.join(os.environ['AZUREML_MODEL_DIR'], model_filename)
    net = Net()
    net.load_state_dict(torch.load(model_path))
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def run(data):
    data = json.loads(data)
    images = torch.FloatTensor(data['data'])
    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    result = [classes[predicted[j]] for j in range(4)]
    result_json = json.dumps({"predictions": result})

    # You can return any JSON-serializable object.
    return result_json
