from collections import namedtuple
import torch
import torch.nn  as nn
import torch.nn.functional as F
import math
import numpy as np


KernelParams = namedtuple('KernelParams',
    ['outchannels', 'shape', 'stride', 'padding'],
    defaults=[1, (3, 3), (1, 1), (0, 0)]
)

PoolingParams = namedtuple('PoolingParams',
    ['shape', 'stride', 'padding', 'mode'],
    defaults=[(3, 3), (1, 1), (0,0), 'max']
)


def _create_conv2d(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)


def _create_pool(kernel_size, stride, padding, mode):
    if mode == 'avg':
        return nn.AvgPool2d(kernel_size, stride, padding)
    elif mode == 'max':
        return nn.MaxPool2d(kernel_size, stride)
    elif mode == 'adapt':
        return nn.AdaptiveAvgPool2d(kernel_size)
    else:
        raise ValueError('Unrecognised pooling mode {}'.format(mode))


def _out_features(*args):
    # expand params
    params = [p if isinstance(p, tuple) else (p,p) for p in args]

    hval, wval = zip(*params)

    # new output shape
    hout = math.floor((hval[0] -  hval[1] + 2 * hval[3]) / hval[2]) + 1
    hout = math.floor((hout - hval[4] + 2 * hval[6]) / hval[5]) + 1

    wout = math.floor((wval[0] -  wval[1] + 2 * wval[3]) / wval[2]) + 1
    wout = math.floor((wout - wval[4] + 2 * wval[6]) / wval[5]) + 1

    return hout, wout


def _conv2D_out_features(input_shape, kernels, pools):
    out_chann, hin, win = input_shape
    out_shape = hin, win

    for k, p in zip(kernels, pools):
        kout, ksize, kstride, kpad = k
        psize, pstride, ppad, _ = p

        out_chann = kout
        out_shape = _out_features(out_shape, ksize, kstride, kpad, psize, pstride, ppad)

    return out_chann, out_shape[0], out_shape[1]


def _unflatten(batch, shape):
    channels, w, h = shape
    return batch.view(-1, channels, w, h)


def _flatten(batch, size):
    return batch.view(-1, size)


class CNN(nn.Module):
    def __init__(self, input_shape, kernels, pools, n_units):
        super(CNN, self).__init__()
        if len(pools) != len(kernels):
            raise ValueError('Number of pooling  and convolution layers do not match')

        if not isinstance(n_units, list):
            n_units = [n_units]

        self.input_shape = input_shape
        self.n_conv_layers = len(kernels)
        self.n_fc_layers = len(n_units)

        in_chann, h, w = input_shape

        conv_layers = []
        for i, (conv_shape, pool_shape) in enumerate(zip(kernels, pools)):

            conv = _create_conv2d(in_chann, *conv_shape)
            pool = _create_pool(*pool_shape)

            cname = 'conv{}'.format(i)
            pname = 'pool{}'.format(i)

            setattr(self, cname, conv)
            setattr(self, pname, pool)

            conv_layers.extend([cname, pname])

            in_chann = conv_shape.outchannels


        in_shape = np.prod(_conv2D_out_features(self.input_shape, kernels, pools))
        self.fc_in_shape = in_shape

        fc_layers = []
        for i, size in enumerate(n_units):
            fc = nn.Linear(in_shape, size)
            in_shape = size

            name = 'fc{}'.format(i)
            setattr(self, name, fc)

            fc_layers.append(name)

        self._conv_layers = conv_layers
        self._fc_layers = fc_layers

    @property
    def conv_layers(self):
        return [getattr(self, name) for name in self._conv_layers]

    @property
    def fc_layers(self):
        return [getattr(self, name) for name in self._fc_layers]

    def forward(self, inputs):
        output = _unflatten(inputs, self.input_shape)

        conv_layers = self.conv_layers
        fc_lauyers = self.fc_layers

        for l in range(self.n_conv_layers):
            conv, pool = conv_layers[2*l: 2*(l+1)]

            output = F.relu(conv(output))
            output = pool(output)


        output = _flatten(output, self.fc_in_shape)

        for fc in fc_lauyers:
            output = fc(output)

        return output

class RCNN(nn.Module):
    def __init__(
        self,
        # CNN params
        input_shape,
        kernels,
        pools,
        # RNN core params
        rnn_core,
        hidden_size=32,
        n_layers=1,
        dropout=0.0,
        batch_first=True
    ):
        super(RCNN, self).__init__()

        if len(pools) != len(kernels):
            raise ValueError('Number of pooling  and convolution layers do not match')

        self.input_shape = input_shape
        self.batch_first = batch_first

        in_chann, h, w = input_shape

        conv_layers = []
        for i, (conv_shape, pool_shape) in enumerate(zip(kernels, pools)):

            conv = _create_conv2d(in_chann, *conv_shape)
            pool = _create_pool(*pool_shape)

            conv_layers.extend([conv, pool])

            in_chann = conv_shape.outchannels

        in_shape = np.prod(_conv2D_out_features(self.input_shape, kernels, pools))

        rnn = init_model(rnn_core, hidden_size, in_shape, n_layers, dropout, False)

        self.cnn = nn.Sequential(*conv_layers)
        self.rnn = rnn

    def reset_parameters(self):
        self.cnn.reset_parameters()
        self.rnn.reset_parameters()

    def forward(self, input, hidden=None):
        if self.batch_first:
            input = input.transpose(0, 1)
        seqlen, batch_size = input.size(0), input.size(1)

        img_features = []
        for img in input:
            unflattened = _unflatten(img, self.input_shape)
            convolved = _flatten(self.cnn(unflattened), batch_size)
            img_features.append(convolved)

        img_features = torch.stack(img_features)

        out, hidden = self.rnn(img_features, hidden)

        if self.batch_first:
            out = out.transpose(0, 1)

        return out, hidden


def init_rcnn(model_type, hidden_size, input_size, n_layers, dropout=0.0, batch_first=True):
    if model_type == 'RNN':
        rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            nonlinearity='relu',
            batch_first=batch_first,
            dropout=dropout
        )

    if model_type == 'LSTM':
        rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=batch_first,
            dropout=dropout
        )

    elif model_type == 'GRU':
        rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=batch_first,
            dropout=dropout
        )

    return rnn


def LeNet(nclasses=10):
    kernels = [
        KernelParams(outchannels=20, shape=(5, 5), stride=(1, 1), padding=(1, 1)),
        KernelParams(outchannels=20, shape=(5, 5), stride=(1, 1), padding=(1, 1)),
    ]

    pools = [
        PoolingParams(shape=(2, 2), stride=(1, 1), padding=(0, 0), mode='max'),
        PoolingParams(shape=(2, 2), stride=(1, 1), padding=(0, 0), mode='max')
    ]

    cnn = CNN((1, 28, 28), kernels, pools, n_units=[500, 500])

    return nn.Sequential(cnn, nn.Linear(500, nclasses), nn.Softmax(dim=1))

def AlexNet(nclasses=10):
    kernels = [
        KernelParams(outchannels=20, shape=(5, 5), stride=(1, 1), padding=(1, 1)),
        KernelParams(outchannels=20, shape=(5, 5), stride=(1, 1), padding=(1, 1)),
    ]

    pools = [
        PoolingParams(shape=(2, 2), stride=(1, 1), padding=(0, 0), mode='max'),
        PoolingParams(shape=(2, 2), stride=(1, 1), padding=(0, 0), mode='max')
    ]

    cnn = CNN((1, 28, 28), kernels, pools, n_units=[500, 500])

    return nn.Sequential(cnn, nn.Linear(500, nclasses), nn.Softmax(dim=1))
