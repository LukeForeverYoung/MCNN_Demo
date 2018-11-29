import torch
import torch.nn as nn

def npToTensor(x,is_cuda=True,requires_grad=False,dtype=torch.FloatTensor):
    if isinstance(x,torch.Tensor):
        t=x.type(dtype)
    else:
        t = torch.from_numpy(x).type(dtype)
    t.requires_grad=requires_grad
    if(is_cuda):
        t=t.cuda()
    return t

class CrowdCounter(nn.Module):
    def __init__(self):
        super(CrowdCounter, self).__init__()
        self.DME = MCNN()
        self.loss_fn = nn.MSELoss()

    @property
    def loss(self):
        return self.loss_mse

    def forward(self, im_data, gt_data=None):
        im_data = npToTensor(im_data, is_cuda=True, requires_grad=self.training)
        density_map = self.DME(im_data)

        if self.training:
            gt_data = npToTensor(gt_data, is_cuda=True, requires_grad=self.training)
            self.loss_mse = self.build_loss(density_map, gt_data)

        return density_map

    def build_loss(self, density_map, gt_data):
        loss = self.loss_fn(density_map, gt_data)
        return loss


class MCNN(nn.Module):
    '''
    Multi-column CNN
        -Implementation of Single Image Crowd Counting via Multi-column CNN (Zhang et al.)
    '''

    def __init__(self):
        super(MCNN, self).__init__()

        self.branch1 = nn.Sequential(Conv2d(1, 16, 9, same_padding=True),
                                     nn.MaxPool2d(2),
                                     Conv2d(16, 32, 7, same_padding=True),
                                     nn.MaxPool2d(2),
                                     Conv2d(32, 16, 7, same_padding=True),
                                     Conv2d(16, 8, 7, same_padding=True))

        self.branch2 = nn.Sequential(Conv2d(1, 20, 7, same_padding=True),
                                     nn.MaxPool2d(2),
                                     Conv2d(20, 40, 5, same_padding=True),
                                     nn.MaxPool2d(2),
                                     Conv2d(40, 20, 5, same_padding=True),
                                     Conv2d(20, 10, 5, same_padding=True))

        self.branch3 = nn.Sequential(Conv2d(1, 24, 5, same_padding=True),
                                     nn.MaxPool2d(2),
                                     Conv2d(24, 48, 3, same_padding=True),
                                     nn.MaxPool2d(2),
                                     Conv2d(48, 24, 3, same_padding=True),
                                     Conv2d(24, 12, 3, same_padding=True))

        self.fuse = nn.Sequential(Conv2d(30, 1, 1, same_padding=True))

    def forward(self, im_data):
        x1 = self.branch1(im_data)
        x2 = self.branch2(im_data)
        x3 = self.branch3(im_data)
        x = torch.cat((x1, x2, x3), 1)
        x = self.fuse(x)

        return x

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        return x