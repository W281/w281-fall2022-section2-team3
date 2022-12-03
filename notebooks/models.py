import torchvision.models as models
from torch import nn

class ResNet152(nn.Module):
    def __init__(self, progress=True):
        super(ResNet152, self).__init__()
        
        # load the pretrained model
        self.model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT, progress=progress)

        # select till the last layer
        # Dropping output layer (the ImageNet classifier)
        self.model = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x):

      x = self.model(x)
      return x


class LogisticRegression(nn.Module):
     def __init__(self, input_dim, output_dim):
         super(LogisticRegression, self).__init__()
         self.model = nn.Linear(input_dim, output_dim)

     def forward(self, x):
         outputs = torch.sigmoid(self.model(x))
         return outputs