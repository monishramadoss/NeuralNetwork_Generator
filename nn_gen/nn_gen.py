import os
import sys
import inspect
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.getcwd()
model_file = open(os.path.join(ROOT,'gen_model.py'), 'w')
model_imports = [
    'import torch\n',
    'import torch.nn as nn\n',
    'import torch.nn.functional as F\n',
    '\n', '\n'
]
model_init = [
    'class Generated_Net(nn.Module):\n',
    '   def __init__(self):\n'
    '       super(Generated_Net, self).__init__()\n',
]
model_forward = ['   def forward(self, x):\n']
model_output = ['       return x\n']

Black_list = ['Container']
LAYER = [m[0] for m in inspect.getmembers(nn, inspect.isclass)]
FUNCTIONAL = ['max_pool2d', 'relu']
layer = []
layer_dict = {}


for i, module in enumerate([m[1] for m in inspect.getmembers(nn, inspect.isclass)]):
    t = inspect.signature(module).parameters.keys()
    if (LAYER[i] not in Black_list):
        layer_dict[LAYER[i]] = list(t)
        layer.append(LAYER[i])

network_layer = '       self.layer{} = nn.{}({})\n'
tab1 = '  '

layerCount = 0
network = []
forward = []
network.append(network_layer.format(layerCount, 'Conv2d', '1, 6, 3'))
layerCount += 1
network.append(network_layer.format(layerCount, 'Conv2d', '6, 16, 3'))
layerCount += 1
network.append(network_layer.format(layerCount, 'Linear', '16 * 6 * 6, 120'))
layerCount += 1
network.append(network_layer.format(layerCount, 'Linear', '120, 84'))
layerCount += 1
network.append(network_layer.format(layerCount, 'Linear', '84, 10'))


model_file.writelines(model_imports)
model_file.writelines(model_init)
network.append('\n')
model_file.writelines(network)
model_file.writelines(model_forward)
model_file.writelines(forward)
model_file.writelines(model_output)
