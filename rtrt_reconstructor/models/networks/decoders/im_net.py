import torch
import torch.nn as nn


class IMNet(nn.Module):
    """
    ref : https://github.com/czq142857/IM-NET-pytorch/blob/master/modelAE.py

    >>> net = IMNet()
    >>> net(torch.randn((10, 1, 3)), torch.randn((10, 256))).shape
    torch.Size([10, 1, 1])
    """
    def __init__(self,
                 z_dim: int = 256,
                 point_dim: int = 3,
                 hidden_dim: int = 128) -> None:
        """
        :param z_dim: # of dim of feature vectors, default 256
        :param point_dim: # of dim of coordinates (3 if 3-d), default 3
        :param hidden_dim: # of hidden dim, default 128
        """
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim + point_dim,
                      hidden_dim * 8,
                      bias=True),
            nn.LeakyReLU(negative_slope=0.02,
                         inplace=True),
            nn.Linear(hidden_dim * 8,
                      hidden_dim * 8,
                      bias=True),
            nn.LeakyReLU(negative_slope=0.02,
                         inplace=True),
            nn.Linear(hidden_dim * 8,
                      hidden_dim * 8,
                      bias=True),
            nn.LeakyReLU(negative_slope=0.02,
                         inplace=True),
            nn.Linear(hidden_dim * 8,
                      hidden_dim * 4,
                      bias=True),
            nn.LeakyReLU(negative_slope=0.02,
                         inplace=True),
            nn.Linear(hidden_dim * 4,
                      hidden_dim * 2,
                      bias=True),
            nn.LeakyReLU(negative_slope=0.02,
                         inplace=True),
            nn.Linear(hidden_dim * 2,
                      hidden_dim * 1,
                      bias=True),
            nn.LeakyReLU(negative_slope=0.02,
                         inplace=True),
            nn.Linear(hidden_dim * 1,
                      1,
                      bias=True))
        self.net.apply(self.init_weights)

    @staticmethod
    def init_weights(m) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0., std=0.02)
            nn.init.constant_(m.bias, 0.)

    def forward(self,
                points: torch.Tensor,
                z: torch.Tensor) -> torch.Tensor:
        """
        :param points: coordinates
        :param z: feature vector
        :return:
        """
        zs = z.view(-1, 1, self.z_dim).repeat(1, points.size()[1], 1)
        inp = torch.cat([points, zs], dim=2)
        out = self.net(inp)
        pred = torch.max(torch.min(out, out * 0.01 + 0.99), out * 0.01)
        return pred


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True)
