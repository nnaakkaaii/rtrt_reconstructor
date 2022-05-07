import torch
import torch.nn as nn


class CNN3D(nn.Module):
    """
    input shape must be (bsize, nch, 64, 64, 64)

    >>> net = CNN3D()
    >>> net(torch.randn((10, 1, 64, 64, 64))).shape
    torch.Size([10, 256])
    """
    def __init__(self,
                 in_dim: int = 1,
                 z_dim: int = 256,
                 hidden_dim: int = 32) -> None:
        """
        :param in_dim: # of dim of input channels, default 1
        :param z_dim: # of dim of feature vectors, default 256
        :param hidden_dim: # of hidden dim, default 32
        """
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Conv3d(in_dim,
                      hidden_dim,
                      kernel_size=(4, 4, 4),
                      stride=(2, 2, 2),
                      padding=1,
                      bias=False),
            nn.InstanceNorm3d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.02,
                         inplace=True),
            nn.Conv3d(hidden_dim,
                      hidden_dim * 2,
                      kernel_size=(4, 4, 4),
                      stride=(2, 2, 2),
                      padding=1,
                      bias=False),
            nn.InstanceNorm3d(hidden_dim * 2),
            nn.LeakyReLU(negative_slope=0.02,
                         inplace=True),
            nn.Conv3d(hidden_dim * 2,
                      hidden_dim * 4,
                      kernel_size=(4, 4, 4),
                      stride=(2, 2, 2),
                      padding=1,
                      bias=False),
            nn.InstanceNorm3d(hidden_dim * 4),
            nn.LeakyReLU(negative_slope=0.02,
                         inplace=True),
            nn.Conv3d(hidden_dim * 4,
                      hidden_dim * 8,
                      kernel_size=(4, 4, 4),
                      stride=(2, 2, 2),
                      padding=1,
                      bias=False),
            nn.InstanceNorm3d(hidden_dim * 8),
            nn.LeakyReLU(negative_slope=0.02,
                         inplace=True))
        self.last_layer = nn.Conv3d(
            hidden_dim * 8,
            z_dim,
            kernel_size=(4, 4, 4),
            stride=(1, 1, 1),
            padding=0,
            bias=True)
        self.net.apply(self.init_weights)
        nn.init.xavier_normal_(self.last_layer.weight)
        nn.init.constant_(self.last_layer.bias, 0.)

    @staticmethod
    def init_weights(m) -> None:
        if isinstance(m, nn.Conv3d):
            nn.init.xavier_normal_(m.weight)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        h1 = self.last_layer(self.net(x))
        h2 = h1.view(-1, self.z_dim)
        pred = torch.sigmoid(h2)
        return pred


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True)
