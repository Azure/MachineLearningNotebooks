'''
Endpoint names to look for in the graph
'''

from anchors import generate_anchors

feat_layers = generate_anchors.feat_layers
sub_feats = ['']
localizations_names = [f'ssd_300_vgg/{feature}_box/Reshape:0' for feature in feat_layers]

predictions_names = ['ssd_300_vgg/softmax/Reshape_1:0'] \
    + [f'ssd_300_vgg/softmax_{n}/Reshape_1:0' for n in range(1, len(feat_layers))]

logit_names = [f'ssd_300_vgg/{feature}_box/Reshape_1:0' for feature in feat_layers]

endpoint_names = ['ssd_300_vgg/conv1/conv1_2/Relu:0'] \
    + [f'ssd_300_vgg/conv{n}/conv{n}_3/Relu:0' for n in range(4, 6)] \
    + [f'ssd_300_vgg/conv{n}/conv{n}_{n}/Relu:0' for n in range(2, 4)] \
    + [f'ssd_300_vgg/conv{n}/Relu:0' for n in range(6, 8)] \
    + [f'ssd_300_vgg/{feature}/conv3x3/Relu:0' for feature in feat_layers if feature != 'block4' and feature != 'block7']