import torch.nn as nn
import math
import torch

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batchSize, channelNum, _, _ = x.size()
        y = self.avg_pool(x)
        y = y.view(batchSize, channelNum)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        y = y.view(batchSize, channelNum, 1, 1)
        return x * y


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class newBottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(newBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn21 = nn.BatchNorm2d(planes)
        self.bn22 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * self.expansion)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn21(out)
        out = self.relu(out)
        if( self.stride == 1):
            out = self.conv2(out)
            out = self.bn22(out)
            out = self.relu(out)
        

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
		
class downUpSample(nn.Module):
    def __init__(self, inplane, midplane):
        super(downUpSample, self).__init__()
        downSample = nn.Sequential(
                conv1x1(inplane, midplane, stride=1),
                nn.BatchNorm2d(midplane),
            )
        upSample = nn.Sequential(
                conv1x1(midplane, inplane, stride=1),
                nn.BatchNorm2d(midplane),
            )
    def forward(self, x):
        x = downSample(x)
        x = upSample(x)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.downUpSample1 = downUpSample(256, 32)
    
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.downUpSample2 = downUpSample(512, 64)
        
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.downUpSample3 = downUpSample(1024, 128)
        
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        
        
    def frezzeFromShallowToDeep(self, lastLayer):
        #conv1 0, layer1 1, layer2 2, layer3 3, layer4 4
        for i, para in enumerate(self.parameters()):
            para.requires_grad = True
        if(lastLayer >= 0):
             for i, para in enumerate(self.conv1.parameters() ):
                para.requires_grad = False
             for i, para in enumerate(self.bn1.parameters() ):
                para.requires_grad = False

        if(lastLayer >= 1):
             for i, para in enumerate(self.layer1.parameters() ):
                para.requires_grad = False

        if(lastLayer >= 2):
             for i, para in enumerate(self.layer2.parameters() ):
                para.requires_grad = False

        if(lastLayer >= 3):
             for i, para in enumerate(self.layer3.parameters() ):
                para.requires_grad = False

        if(lastLayer >= 4):
             for i, para in enumerate(self.layer4.parameters() ):
                para.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
       

        x = self.layer1(x)
        x = self.downUpSample1(x)
        
        x = self.layer2(x)
        x = self.downUpSample2(x)
        
        x = self.layer3(x)
        x = self.downUpSample3(x)
        
        x = self.layer4(x)


        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
              
        
        return x

    
    def getParameters(self, featurePos):
        result_params = None
        
        if(featurePos == 0):
            conv_params = list(map(id, self.conv1.parameters()))
            bn_params = list(map(id, self.bn1.parameters()))
            result_params = filter(lambda p: id(p) in conv_params + bn_params , self.parameters())
        elif(featurePos == 1):
            layer1_params = list(map(id, self.layer1.parameters()))
            result_params = filter(lambda p: id(p) in layer1_params, self.parameters())
            
        elif(featurePos == 2):
            layer2_params = list(map(id, self.layer2.parameters()))
            result_params = filter(lambda p: id(p) in layer2_params, self.parameters())
            
        elif(featurePos == 3):
            layer3_params = list(map(id, self.layer3.parameters()))
            result_params = filter(lambda p: id(p) in layer3_params, self.parameters())
            
        elif(featurePos == 4):
            layer4_params = list(map(id, self.layer4.parameters()))
            result_params = filter(lambda p: id(p) in layer4_params, self.parameters())
            
        elif(featurePos == 5):
            fc_params = list(map(id, self.fc.parameters()))
            result_params = filter(lambda p: id(p) in fc_params, self.parameters())
         
        return result_params
    
def se_resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pass
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

def se_resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        pass
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

class simpleFcNet(nn.Module):
    def __init__(self, num_classes):
        super(simpleFcNet, self).__init__()
        self.fc = nn.Linear(num_classes, num_classes)
        self.initialize()
        
    def initialize(self):
        nn.init.eye_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        out = x
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def se_resnet(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pass
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model