import argparse
import numpy as np
import torch
from torch import nn, utils, optim
from torchvision import transforms, models
import torch.nn.functional as F
from PIL import Image

class ConversionNet(nn.Module):
    def __init__(self):
        super(ConversionNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3, bias=False)
        self.norm1 = nn.InstanceNorm2d(num_features=64, affine=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, bias=False)
        self.norm2 = nn.InstanceNorm2d(num_features=128, affine=True)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm3 = nn.InstanceNorm2d(num_features=128, affine=True)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm4 = nn.InstanceNorm2d(num_features=128, affine=True)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm5 = nn.InstanceNorm2d(num_features=128, affine=True)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.norm6 = nn.InstanceNorm2d(num_features=64, affine=True)
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=3, bias=False)

    def forward(self, input):
        x = self.conv1(input)
        x = F.relu(self.norm1(x))
        x = self.conv2(x)
        x = F.relu(self.norm2(x))
        s = x
        x = self.conv3(x)
        x = F.relu(self.norm3(x))
        x = self.conv4(x)
        x = F.relu(self.norm4(x))
        x = self.conv5(x)
        x = F.relu(self.norm5(x))
        x = x + s
        x = F.interpolate(x, scale_factor=2)
        x = self.conv6(x)
        x = F.relu(self.norm6(x))
        x = torch.tanh(self.conv7(x))
        output = torch.clip(input + x, -1.0, 1.0)
        return output

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='', help='input image file.')
    parser.add_argument('--output_file', type=str, default='', help='result image file to write.')
    parser.add_argument('--gpu', default=0, type=int, help='visible gpu number.')
    args = parser.parse_args()

    if args.gpu >= 0:
        device = torch.device("cuda:%d"%args.gpu)
    else:
        device = torch.device("cpu")

    model = ConversionNet()
    model.load_state_dict(torch.load("Anime.pt"))
    model.to(device)

    im = Image.open(args.input_file)

    a = (np.array(im.resize((256,256)), dtype=np.float32) / 127.5) - 1.
    t = a.transpose((2,0,1)).reshape((1,3,256,256))
    t = torch.Tensor(t).to(device)

    p = model(t).detach().cpu().numpy()
    p = p.reshape((3,256,256)).transpose((1,2,0))
    p = ((p*0.5 + 0.5) * 255).astype(np.uint8)

    d = Image.fromarray(p)
    d.resize((im.width,im.height))
    d.save(args.output_file)


if __name__ == '__main__':
    main()
