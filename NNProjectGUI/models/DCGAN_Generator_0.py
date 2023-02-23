import torch.nn as nn

class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.main = nn.Sequential(
            #input 100 * 1024 * 3

            nn.ConvTranspose2d(100, 1024,kernel_size= 4, stride= 2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            #output  3 * 3

            nn.ConvTranspose2d(1024, 512, kernel_size= 4, stride=2,padding= 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #output  7*7

            nn.ConvTranspose2d(512, 256, kernel_size= 4, stride=2,padding= 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            #output  14 * 14

            nn.ConvTranspose2d(256, 128, kernel_size= 4, stride=2,padding= 1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # output  28*28

            nn.ConvTranspose2d(128, 64, kernel_size= 4, stride=2,padding= 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, kernel_size= 4, stride=2,padding= 1, bias=False),
            nn.Tanh()
            )

    def forward(self, input):
        return self.main(input)