import torch.nn.functional as F
import torch.nn as nn
import torch
import segmentation_models_pytorch as smp

class Cloud2Cloud(nn.Module):
    def __init__(self, in_channels=24):
        super(Cloud2Cloud, self).__init__()

        self.cloudNet = smp.PSPNet(encoder_name='resnet34', classes=4, encoder_weights='imagenet')
        self.cloudNet.encoder.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(5,5), stride=(2,2), padding=(2,2), bias=False)
        self.metNet= smp.PSPNet(encoder_name='resnet18', encoder_weights='imagenet')
        self.cloud_encoder = self.cloudNet.encoder
        self.met_encoder = self.metNet.encoder  
        self.cloud_decoder = self.cloudNet.decoder

    def forward(self, inputs):

        cloud_z = self.cloud_encoder(inputs[:,:in_channels,:,:])
        met_z = self.met_encoder(inputs[:,in_channels:,:,:])
        cloud_z[0] = met_z[0]+cloud_z[0]        
        output = self.cloud_decoder(cloud_z)
        
        return output