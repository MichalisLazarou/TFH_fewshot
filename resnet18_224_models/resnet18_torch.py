import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlockDec(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        super(BasicBlockDec, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.sgm = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.upsample = upsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.planes = planes


    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x
        #print(x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            out = nn.functional.interpolate(out,scale_factor=2, mode='bilinear', align_corners=True)
            residual = self.upsample(x)
        out += residual
        if self.planes ==3:
            out = self.sgm(out)
        else:
            out = self.relu(out)
        #out = self.maxpool(out)

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)
        return out

class ResNetDec(nn.Module):

    def __init__(self, args,block=BasicBlockDec, keep_prob=1.0, avg_pool=False, drop_rate=0.0, dropblock_size=5):
        self.inplanes = 3
        super(ResNetDec, self).__init__()
        self.image_size = args.image_size
        if args.image_size>84:
            self.layer0 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2)
            self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
            self.layer2 = self._make_layer(block, 128, stride=2, drop_rate=drop_rate)
            self.layer3 = self._make_layer(block, 256, stride=2, drop_rate=drop_rate, drop_block=True,block_size=dropblock_size)
            self.layer4 = self._make_layer(block, 512, stride=2, drop_rate=drop_rate, drop_block=True,block_size=dropblock_size)
        else:
            self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
            self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
            self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate, drop_block=True,block_size=dropblock_size)
            self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        if avg_pool:
            self.avgpool = nn.AvgPool2d(5, stride=1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        #print(planes, self.inplanes)
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(planes * block.expansion, self.inplanes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.inplanes),
            )

        layers = []
        layers.append(block(planes, self.inplanes, stride, downsample, drop_rate, drop_block, block_size))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        #print(x.shape)
        x = self.layer4(x)
        #print(x.shape)

        x = self.layer3(x)
        #print(x.shape)

        x = self.layer2(x)
        #print(x.shape)
        if self.image_size ==224:
            x = self.layer0(x)
            #print(x.shape)
        x = self.layer1(x)

        x = nn.functional.interpolate(x, size= self.image_size, mode = 'bilinear', align_corners = True)
        return x

# code taken from https://github.com/tankche1/IDeMe-Net/blob/master/classification.py
class ClassificationNetwork(nn.Module):
    def __init__(self, params):
        super(ClassificationNetwork, self).__init__()
        self.convnet = torchvision.models.resnet18(pretrained=False)
        num_ftrs = self.convnet.fc.in_features
        self.device = params.device
        self.dataset = params.dataset
        self.convnet.fc = nn.Linear(num_ftrs, params.num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs):
        outputs = self.convnet(inputs)
        return outputs

    def normalize(self, datas):
        norms = datas.norm(dim=1, keepdim=True)
        return datas / norms

    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 100
        avg_loss = 0

        for i, (x, y) in enumerate(train_loader):
            #-----------Problem in training for CUB database
            #getting from 0-198 labels, converting them to 0-99
            if self.dataset == "CUB":
                y = y/2
            x, y = x.to(self.device), y.to(self.device)
            outputs=self.forward(x)
            #print(outputs.shape)
            loss = self.criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss + loss.item()

            if i % print_freq == 0:
                # print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss / float(i + 1)))


# resnet18 without fc layer, used for testing to extract features
class EmbeddingNet(nn.Module):
    def __init__(self, params):
        super(EmbeddingNet, self).__init__()
        self.resnet = ClassificationNetwork(params)
        ckpt = torch.load(params.file_path)
        self.resnet.load_state_dict(ckpt['model_state_dict'])

        self.conv1 = self.resnet.convnet.conv1
        self.conv1.load_state_dict(self.resnet.convnet.conv1.state_dict())
        self.bn1 = self.resnet.convnet.bn1
        self.bn1.load_state_dict(self.resnet.convnet.bn1.state_dict())
        self.relu = self.resnet.convnet.relu
        self.maxpool = self.resnet.convnet.maxpool
        self.layer1 = self.resnet.convnet.layer1
        self.layer1.load_state_dict(self.resnet.convnet.layer1.state_dict())
        self.layer2 = self.resnet.convnet.layer2
        self.layer2.load_state_dict(self.resnet.convnet.layer2.state_dict())
        self.layer3 = self.resnet.convnet.layer3
        self.layer3.load_state_dict(self.resnet.convnet.layer3.state_dict())
        self.layer4 = self.resnet.convnet.layer4
        self.layer4.load_state_dict(self.resnet.convnet.layer4.state_dict())
        self.layer4 = self.resnet.convnet.layer4
        self.layer4.load_state_dict(self.resnet.convnet.layer4.state_dict())
        self.avgpool = self.resnet.convnet.avgpool

    def forward(self,x, feat_tensor = False):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        layer1 = self.layer1(x) # (, 64L, 56L, 56L)
        layer2 = self.layer2(layer1) # (, 128L, 28L, 28L)
        layer3 = self.layer3(layer2) # (, 256L, 14L, 14L)
        layer4 = self.layer4(layer3) # (,512,7,7)
        if feat_tensor:
            return layer4
        else:
            x = self.avgpool(layer4)  # (,512,1,1)
            x = x.view(x.size(0), -1)
        return x
