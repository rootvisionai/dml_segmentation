import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision.models import resnet18
from torchvision.models import resnet34
from torchvision.models import resnet50
from torchvision.models import resnet101

def l2_norm(vector):
    a_norm = vector.norm(dim=-1, p=2)
    vector = vector.divide(a_norm.unsqueeze(1))
    return vector

class EmbSizeReduction(nn.Module):
    def __init__(self, pre_dim=2048, dim=512):
        super().__init__()
        self.pre_dim = pre_dim
        self.dim = dim

        self.reduction_coeff = nn.Sequential(
            nn.Linear(pre_dim, int(pre_dim / dim), bias=False),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = l2_norm(x)
        a = self.reduction_coeff(x)
        x = x.reshape(x.shape[0], int(self.pre_dim / self.dim), self.dim)
        x = x * a[..., None]
        return x.sum(1)

class Resnet18(nn.Module):
    def __init__(self, embedding_size, pretrained=True, is_norm=True, bn_freeze = True,
                 attention=False, attention_heads=4):
        super(Resnet18, self).__init__()

        self.model = resnet18(pretrained)
        self.is_norm = is_norm
        self.embedding_size = embedding_size
        self.num_ftrs = self.model.fc.in_features
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)
        self.model.embedding = nn.Linear(self.num_ftrs, self.embedding_size * attention_heads)

        if attention:
            self.model.attention = EmbSizeReduction(pre_dim = embedding_size * attention_heads, dim = embedding_size)
        else:
            self.model.attention = nn.Identity()
        
        self.classifier = nn.Identity() 

        self._initialize_weights()

        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        avg_x = self.model.gap(x)
        max_x = self.model.gmp(x)

        x = max_x + avg_x
        
        x = x.view(x.size(0), -1)
        x = self.model.embedding(x)
        
        x = self.model.attention(x)
        
        x = self.classifier(x)
            
        return x

    def _initialize_weights(self):
        init.kaiming_normal_(self.model.embedding.weight, mode='fan_out')
        init.constant_(self.model.embedding.bias, 0)


class Resnet34(nn.Module):
    def __init__(self, embedding_size, pretrained=True, is_norm=True, bn_freeze=True,
                 attention=False, attention_heads=4):
        super(Resnet34, self).__init__()

        self.model = resnet34(pretrained)
        self.is_norm = is_norm
        self.embedding_size = embedding_size
        self.num_ftrs = self.model.fc.in_features
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)

        self.model.embedding = nn.Linear(self.num_ftrs, self.embedding_size * attention_heads)

        if attention:
            self.model.attention = EmbSizeReduction(pre_dim = embedding_size * attention_heads, dim = embedding_size)
        else:
            self.model.attention = nn.Identity()

        self.classifier = nn.Identity()
            
        self._initialize_weights()

        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        avg_x = self.model.gap(x)
        max_x = self.model.gmp(x)

        x = max_x + avg_x
        
        x = x.view(x.size(0), -1)
        x = self.model.embedding(x)
        
        x = self.model.attention(x)
        
        x = self.classifier(x)
            
        return x

    def _initialize_weights(self):
        init.kaiming_normal_(self.model.embedding.weight, mode='fan_out')
        init.constant_(self.model.embedding.bias, 0)

    def _initialize_weights(self):
        init.kaiming_normal_(self.model.embedding.weight, mode='fan_out')
        init.constant_(self.model.embedding.bias, 0)


class Resnet50(nn.Module):
    def __init__(self, embedding_size, pretrained=True, is_norm=True, bn_freeze=True,
                 attention=False, attention_heads=4):
        super(Resnet50, self).__init__()

        self.model = resnet50(pretrained)
        self.is_norm = is_norm
        self.embedding_size = embedding_size
        self.num_ftrs = self.model.fc.in_features
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)

        self.model.embedding = nn.Linear(self.num_ftrs, self.embedding_size * attention_heads)

        if attention:
            self.model.attention = EmbSizeReduction(pre_dim = embedding_size * attention_heads, dim = embedding_size)
        else:
            self.model.attention = nn.Identity()

        self.classifier = nn.Identity()
            
        self._initialize_weights()

        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        avg_x = self.model.gap(x)
        max_x = self.model.gmp(x)

        x = max_x + avg_x
        
        x = x.view(x.size(0), -1)
        x = self.model.embedding(x)
        
        x = self.model.attention(x)
            
        x = self.classifier(x)
            
        return x

    def _initialize_weights(self):
        init.kaiming_normal_(self.model.embedding.weight, mode='fan_out')
        init.constant_(self.model.embedding.bias, 0)


class Resnet101(nn.Module):
    def __init__(self, embedding_size, pretrained=True, is_norm=True, bn_freeze=True,
                 attention=False, attention_heads=4):
        super(Resnet101, self).__init__()

        self.model = resnet101(pretrained)
        self.is_norm = is_norm
        self.embedding_size = embedding_size
        self.num_ftrs = self.model.fc.in_features
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)

        self.model.embedding = nn.Linear(self.num_ftrs, self.embedding_size * attention_heads)

        if attention:
            self.model.attention = EmbSizeReduction(pre_dim = embedding_size * attention_heads, dim = embedding_size)
        else:
            self.model.attention = nn.Identity()
            
        self.classifier = nn.Identity()

        self._initialize_weights()

        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        avg_x = self.model.gap(x)
        max_x = self.model.gmp(x)

        x = max_x + avg_x
        
        x = x.view(x.size(0), -1)
        x = self.model.embedding(x)
        
        x = self.model.attention(x)
        
        x = self.classifier(x)
        
        return x

    def _initialize_weights(self):
        init.kaiming_normal_(self.model.embedding.weight, mode='fan_out')
        init.constant_(self.model.embedding.bias, 0)