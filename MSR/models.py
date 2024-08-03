import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import ShareConv2d

class ConvNet(nn.Module):
    def __init__(self, n_way):
        super(ConvNet, self).__init__()
        self.shareconv1 = ShareConv2d(1, 64, 3, bias=False) # de base pas de bias
        self.bn1 = nn.BatchNorm2d(64, momentum=1, affine=True)
        self.shareconv2 = ShareConv2d(64, 64, 3, bias=False) # de base pas de bias
        self.bn2 = nn.BatchNorm2d(64, momentum=1, affine=True)
        self.shareconv3 = ShareConv2d(64, 64, 3, bias=False) # de base pas de bias
        self.bn3 = nn.BatchNorm2d(64, momentum=1, affine=True)
        self.flatten = Flatten()
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64, 64)  # Assuming the input size to the linear layer
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, n_way)

    def forward(self, x):

        x = self.shareconv1(x)

        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = self.shareconv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = self.shareconv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = self.flatten(x)  # Flatten all dimensions except batch
        # print(f"Shape after flatten: {x.shape}")  # Ajoutez cette ligne pour vérifier la dimension

        x = self.dropout1(x)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        
        return output
    

class ConvNet_Sans_Dropout(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet_Sans_Dropout, self).__init__()
        self.shareconv1 = ShareConv2d(1, 64, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=1, affine=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.shareconv2 = ShareConv2d(64, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=1, affine=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.shareconv3 = ShareConv2d(64, 64, 3, bias=False)
        self.bn3 = nn.BatchNorm2d(64, momentum=1, affine=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.shareconv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.shareconv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.shareconv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.fc(x)

        return x
    

class ConvNet_with_Dropout(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet_with_Dropout, self).__init__()
        self.conv1 = ShareConv2d(1, 64, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=1, affine=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = ShareConv2d(64, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=1, affine=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.2)

        self.conv3 = ShareConv2d(64, 64, 3, bias=False)
        self.bn3 = nn.BatchNorm2d(64, momentum=1, affine=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.flatten(x)
        x = self.fc(x)

        return x

    
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)