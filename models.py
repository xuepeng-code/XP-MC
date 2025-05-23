import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from attention_copy import CrossModalAttention,SelfAttentionMLP
from torchvision.ops import RoIAlign
import pandas as pd
import nibabel as nib
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool
import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, dropout_rate=0.02):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=dropout_rate)  
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)


        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)  

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual


        return out

class ResNet_3D(nn.Module):
    def __init__(self, block, layers, input_D, input_H, input_W, num_classes=128, dropout_rate=0.0):
        self.inplanes = 64
        super(ResNet_3D, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=dropout_rate)  # 添加 Dropout 层
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0], dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, dropout_rate=dropout_rate)

        # Fully connected layer to output a 128-dimensional feature vector
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.dropout_fc = nn.Dropout(p=dropout_rate)  #  Dropout

        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, dropout_rate=0.0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample, dropout_rate=dropout_rate))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, dropout_rate=dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.dropout(x)  # Dropout
        x = self.maxpool(x)
        x = self.layer1(x)
        
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global Average Pooling
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = torch.flatten(x, 1)
        # x = self.dropout_fc(x)  # Dropout
        # Fully connected layer to output a 128-dimensional feature vector
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        """ Initialize weights for all layers. """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                m.bias.data.zero_()
    
class MLP_gene(nn.Module):
    def __init__(self, input_size, hidden1_size=256, hidden2_size=128, output_size=128,dropout_prob=0.7):
        super(MLP_gene, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)  
        self.fc2 = nn.Linear(hidden1_size, hidden2_size) 
        self.fc3 = nn.Linear(hidden2_size, output_size) 
        # self.attention = SelfAttentionMLP(input_size)
        self.fc_project = nn.Linear(output_size, hidden1_size)
        self.fc4 = nn.Linear(hidden1_size, output_size) 
          # input_size
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
    def forward(self, x):

        
        # gene_att_weight,x = self.attention(x)
        x = F.relu(self.fc1(x))  # (batch_size, hidden1_size)
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))  # (batch_size, hidden2_size)
        x = self.dropout2(x)
        x1 = self.fc3(x)  # (batch_size, output_size)


        return x1

    
class MLP_protein(nn.Module):
    def __init__(self, input_size, hidden1_size=128, hidden2_size=128, output_size=128):
        super(MLP_protein, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size)

        self.fc_project = nn.Linear(output_size, hidden1_size)
        self.fc4 = nn.Linear(hidden1_size, output_size)

        

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x1 = self.fc3(x)  

        return x1

class Model_all(nn.Module):
    def __init__(self):  # num_snps,num_protein
        super(Model_all, self).__init__()
        self.shared_fc = nn.Linear(256, 128)
        self.attn12 = CrossModalAttention(feature_dim=128, shared_fc=self.shared_fc)
        self.attn13 = CrossModalAttention(feature_dim=128, shared_fc=self.shared_fc)
        self.attn23 = CrossModalAttention(feature_dim=128, shared_fc=self.shared_fc)
        self.attn11 = CrossModalAttention(feature_dim=128, shared_fc=self.shared_fc)
        # self.attn12 = CrossModalAttention(feature_dim=128)
        # self.attn13 = CrossModalAttention(feature_dim=128)
        # self.attn23 = CrossModalAttention(feature_dim=128)
        # self.attn11 = CrossModalAttention(feature_dim=128)
        self.bn_mri = nn.BatchNorm1d(128)
        self.bn_snp = nn.BatchNorm1d(128)
        self.bn_protein = nn.BatchNorm1d(128)

    def forward(self, mri_data, snp_data, protein_data):
        mri_embedding = mri_data
        snp_output = snp_data
        protein_output = protein_data
        mri_embedding = self.bn_mri(mri_embedding)
        snp_output = self.bn_snp(snp_output)
        protein_output = self.bn_protein(protein_output)
        attended_ms, attn_weights_ms = self.attn12(mri_embedding, snp_output)
        attended_mp, attn_weights_mp = self.attn13(mri_embedding, protein_output)
        attended_sp, attn_weights_sp = self.attn23(snp_output, protein_output)

        return {
            'mri_embedding': mri_embedding,
            'snp_output': snp_output,
            'protein_output': protein_output,            
            'attended_ms': attended_ms,
            'attended_mp': attended_mp,
            'attended_sp': attended_sp,
            'attn_weights_ms': attn_weights_ms,
            'attn_weights_mp': attn_weights_mp,
            'attn_weights_sp': attn_weights_sp
        }

class FeatureProjector(nn.Module):
    def __init__(self, output_size):
        super(FeatureProjector, self).__init__()
        self.fc = nn.Linear(384, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.fc(x))

class Classifier_multi(nn.Module):
    def __init__(self, input_size=128,num_classes=2):
        super(Classifier_multi, self).__init__()
        self.fc1 = nn.Linear(384, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        # x1 = self.softmax(x)
        return x

    
class Classifier_MRI(nn.Module):
    def __init__(self, input_size=134, num_classes=128):
        super(Classifier_MRI, self).__init__()
        # self.attention = SelfAttentionMLP(input_size)
        self.fc1 = nn.Linear(input_size, 128) 
        self.fc2 = nn.Linear(128, 128)        
        self.fc3 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1) 

    def forward(self, x):
        # MRI_att_weight,x = self.attention(x)
        x = self.relu(self.fc1(x))  
        x = self.relu(self.fc2(x))  
        x = self.fc3(x)  
        # x1 = self.softmax(x)         
        return x    
    
     
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, dropout_rate=0.05):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet34_3D(nn.Module):
    def __init__(self, block, layers, input_D, input_H, input_W, num_classes=128, dropout_rate=0.05):
        super(ResNet34_3D, self).__init__()
        self.inplanes = 64

        # First convolutional layer
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0], dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, dropout_rate=dropout_rate)

        # Fully connected layer to output a 128-dimensional feature vector
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.dropout_fc = nn.Dropout(p=dropout_rate)  # Dropout

        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, dropout_rate=0.05):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample, dropout_rate=dropout_rate))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, dropout_rate=dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)  # Dropout
        x = self.maxpool(x)
        x = self.layer1(x)
        
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global Average Pooling
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = torch.flatten(x, 1)
        # x = self.dropout_fc(x)  # Dropout
        # Fully connected layer to output a 128-dimensional feature vector
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        """ Initialize weights for all layers. """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                m.bias.data.zero_()