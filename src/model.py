import os
import torch
import torchvision
from torchvision import transforms

# Model for 5 base channels + 5 means
class face_model(torch.nn.Module):
    def __init__(self, params):
        super(face_model, self).__init__()
                
        self.conversion_layer = torch.nn.Conv2d(
            in_channels=10,
            out_channels=10,
            kernel_size=(3,3),
            padding='same')
        self.conversion_layer_2 = torch.nn.Conv2d(
            in_channels=10,
            out_channels=7,
            kernel_size=(3,3),
            padding='same')
        self.conversion_layer_1 = torch.nn.Conv2d(
            in_channels=7,
            out_channels=3,
            kernel_size=(1,1),
            padding='same')

        self.activation = torch.nn.ReLU()

        self.base_model = torchvision.models.inception_v3(
            weights=torchvision.models.Inception_V3_Weights(
                torchvision.models.Inception_V3_Weights.DEFAULT
            )
        )
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.fc = torch.nn.Identity()
        self.base_model.eval()

        self.embedding_layer = torch.nn.Linear(
            in_features=2048,
            out_features=params.embedding_size
        )

    def forward(self, x):
        x = self.conversion_layer(x)
        x = self.activation(x)
        x = self.conversion_layer_2(x)
        x = self.activation(x)
        x = self.conversion_layer_1(x)
        x = self.activation(x)
        print(x.shape)
        x = self.base_model(x)
        x = torch.flatten(x)
        x = self.embedding_layer(x)
        return x