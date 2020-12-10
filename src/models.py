import torch
import torch.nn as nn
import torch.nn.functional as F


def get_model(net_name, **kwargs):
    if net_name=="ConvNet":
        return ConvNet(**kwargs)
    elif net_name == "ResNet":
        return ResNet(**kwargs)
    elif net_name == "LeNet":
        return LeNet()
    elif net_name == "FCN":
        return simple_network()
    else:
        raise NotImplementedError("{} is not implemented".format(net_name))


class ConvBlock(nn.Module):

    def __init__(self, in_chan, out_chan, kernel_size, bias, stride, padding, act_fn):
        super().__init__()
        self.model = nn.Sequential()
        self.model.add_module(
                "Conv2d",
                nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=kernel_size,
                      stride=stride, padding=padding)
        )

        if act_fn is not None:
            self.model.add_module(act_fn, getattr(nn, act_fn)())

        self.model.add_module(
                "BatchNorm",
                nn.BatchNorm2d(out_chan)
                )

    def forward(self, x):
        return self.model(x)


class CLayers(nn.Module):

    def __init__(self, in_chan, scale_factor, num_layer, kernel_size, act_fn, maxpool_freq, bias, base):
        super().__init__()
        self.model = nn.Sequential()
        out_chan = base
        for now in range(num_layer):
            self.model.add_module(
                    "ConvBlock:{}".format(now),
                    ConvBlock(in_chan=in_chan, out_chan=out_chan,kernel_size=kernel_size,
                        bias=bias, stride=1, padding=kernel_size//2, act_fn=act_fn)
            )
            in_chan = out_chan
            out_chan = int(in_chan * scale_factor)
            if now % maxpool_freq == maxpool_freq - 1:
                self.model.add_module(
                        "Maxpool:{}".format(now),
                        nn.MaxPool2d(2,2)
                        )

    def forward(self, x):
        return self.model(x)


class ResBlock(nn.Module):

    def __init__(self, in_chan, kernel_size, bias, stride, padding, base, act_fn):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chan, base, kernel_size, stride, padding, bias=bias)
        self.actfn = getattr(nn, act_fn)()
        self.conv2 = nn.Conv2d(base, base, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(base)

    def forward(self, x):
        out = self.conv1(x)
        out = self.actfn(out)
        out = self.bn(out)
        out = self.conv2(out)
        out = self.actfn(out)
        return out + x


class ResStack(nn.Module):

    def __init__(self, in_chan, num_layer, kernel_size, act_fn, maxpool_freq, base, bias):
        super().__init__()
        self.in_conv = nn.Conv2d(in_chan, base, 1, 1, 1,)
        self.model = nn.Sequential()
        for now in range(num_layer):
            self.model.add_module(
                    "ResBlock:{}".format(now),
                    ResBlock(base, kernel_size, bias, 1, kernel_size//2, base, act_fn)
                    )
            if now % maxpool_freq == maxpool_freq - 2:
                self.model.add_module(
                        "MacPool:{}".format(now),
                        nn.MaxPool2d(2,2)
                        )

    def forward(self, x):
        x = self.in_conv(x)
        return self.model(x)


class LinearLayers(nn.Module):

    def __init__(self, in_chan, output, scale_factor, num_layer, act_fn, final):
        super().__init__()
        self.model = nn.Sequential()
        for now in range(num_layer):
            out_chan = int(in_chan * scale_factor)
            if now == num_layer - 1:
                out_chan = output
            self.model.add_module(
                    "Liear:{}".format(now),
                    nn.Linear(in_chan, out_chan)
            )
            if now + 1  == num_layer:
                act_fn = final
            if act_fn is not None:
                self.model.add_module(act_fn, getattr(nn, act_fn)())
            in_chan = out_chan

    def forward(self, x):
        return self.model(x)


class ConvNet(nn.Module):

    def __init__(self, in_chan, scale_factor, num_layer, kernel_size, act_fn, maxpool_freq,
            bias, linear_in, linear_num, l_scale_factor, out_chan, base, final):
        super().__init__()
        self.cnn = CLayers(in_chan=in_chan, scale_factor=scale_factor, num_layer=num_layer, bias=bias,
                kernel_size=kernel_size, act_fn=act_fn, maxpool_freq=maxpool_freq, base=base)
        self.linear = LinearLayers(in_chan=linear_in, output=out_chan, scale_factor=l_scale_factor,
                num_layer=linear_num, act_fn=act_fn, final=final)

    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x,1)
        # print(x.size())
        return self.linear(x)


class ResNet(nn.Module):

    def __init__(self, in_chan, base, num_layer, kernel_size, act_fn, maxpool_freq,
            bias, linear_in, linear_num, scale_factor, out_chan, final):
        super().__init__()
        self.resblock = ResStack(in_chan, num_layer, kernel_size, act_fn, maxpool_freq, base, bias)
        self.linear = LinearLayers(linear_in, out_chan, scale_factor, linear_num, act_fn, final)

    def forward(self, x):
        x = self.resblock(x)
        x = torch.flatten(x, 1)
        # print(x.size())
        return self.linear(x)


class LeNet(nn.Module):
    def __init__(self, input_dim=1, num_class=10):
        super(LeNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_dim, 20,  kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20,    50,  kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(50)

        # Fully connected layers
        self.fc1 = nn.Linear(800, 500)
        #self.bn3 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, num_class)

        # Activation func.
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))                                  #  28 x 28 x 1 -> 24 x 24 x 20
        x = self.bn1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # 12 x 12 x 20
        x = self.relu(self.conv2(x))                                   # -> 8 x 8 x 50
        x = self.bn2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # -> 4 x 4 x 50

        b,c,h,w = x.size()                                                   # batch, channels, height, width
        x = x.view(b, -1)                                                    # flatten the tensor x -> 800

        x = self.relu(self.fc1(x))          # fc-> ReLU
        x = self.fc2(x)                           # fc
        return x


class simple_network(nn.Module):
    def __init__(self, input_dim=1, num_class=10):
        super(simple_network, self).__init__()

        # Fully connected layers
        self.fc1 = nn.Linear(784, 512)
        #nn.init.kaiming_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(512, num_class)
        #nn.init.kaiming_uniform_(self.fc2.weight)

        # Activation func.
        self.relu = nn.ReLU()

        #self.dropout1 = nn.Dropout(0.2)

    def forward(self, x):

        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        # print("Trainig Data is limited, only the first "+str(self.data.size(0))+" samples are used.")
        x = self.relu(self.fc2(x))
        # x = F.relu(self.fc1(x))
        # x = self.dropout1(x)
        # x = F.relu(self.fc2(x))

        return x
