import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int) -> None:
        super().__init__()
        if in_dim == out_dim:
            self.in_net = nn.Sequential()
            self.out_net = nn.Sequential(
                nn.Conv2d(in_dim,
                          out_dim,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1,
                          bias=False),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(negative_slope=0.02,
                             inplace=True),
                nn.Conv2d(out_dim,
                          out_dim,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1,
                          bias=False),
                nn.BatchNorm2d(out_dim))
        else:
            self.in_net = nn.Sequential(
                nn.Conv2d(in_dim,
                          out_dim,
                          kernel_size=(1, 1),
                          stride=(2, 2),
                          padding=0,
                          bias=False),
                nn.BatchNorm2d(out_dim))
            self.out_net = nn.Sequential(
                nn.Conv2d(in_dim,
                          out_dim,
                          kernel_size=(3, 3),
                          stride=(2, 2),
                          padding=1,
                          bias=False),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(negative_slope=0.02,
                             inplace=True),
                nn.Conv2d(out_dim,
                          out_dim,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1,
                          bias=False),
                nn.BatchNorm2d(out_dim))

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        h = self.out_net(x)
        x2 = self.in_net(x)
        h2 = x2 + h
        y = F.leaky_relu(h2,
                         negative_slope=0.02,
                         inplace=True)
        return y


class ResNet2D(nn.Module):
    """
    input shape must be (bsize, nch, 128, 128)

    >>> net = ResNet2D()
    >>> net(torch.randn((10, 1, 128, 128))).shape
    torch.Size([10, 256])
    """
    def __init__(self,
                 in_dim: int = 1,
                 z_dim: int = 256,
                 hidden_dim: int = 64) -> None:
        """
        :param in_dim: # of input channels, default 1
        :param z_dim: # of dim of feature vectors, default 1
        :param hidden_dim: # of hidden dim, default 64
        """
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_dim,
                      hidden_dim,
                      kernel_size=(7, 7),
                      stride=(2, 2),
                      padding=3,
                      bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.02,
                         inplace=True),
            ResNetBlock(hidden_dim,
                        hidden_dim),
            ResNetBlock(hidden_dim,
                        hidden_dim),
            ResNetBlock(hidden_dim,
                        hidden_dim * 2),
            ResNetBlock(hidden_dim * 2,
                        hidden_dim * 2),
            ResNetBlock(hidden_dim * 2,
                        hidden_dim * 4),
            ResNetBlock(hidden_dim * 4,
                        hidden_dim * 4),
            ResNetBlock(hidden_dim * 4,
                        hidden_dim * 8),
            ResNetBlock(hidden_dim * 8,
                        hidden_dim * 8),
            nn.Conv2d(hidden_dim * 8,
                      hidden_dim * 8,
                      kernel_size=(4, 4),
                      stride=(2, 2),
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.LeakyReLU(negative_slope=0.02,
                         inplace=True),
            nn.Conv2d(hidden_dim * 8,
                      z_dim,
                      kernel_size=(4, 4),
                      stride=(1, 1),
                      padding=0,
                      bias=True))
        self.net.apply(self.init_weights)

    @staticmethod
    def init_weights(m) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        h = self.net(1 - x)
        y = h.view(-1, self.z_dim)
        pred = torch.sigmoid(y)
        return pred


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True)
