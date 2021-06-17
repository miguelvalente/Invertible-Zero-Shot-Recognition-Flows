import torch
import pyro.distributions as dist
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils import make_toy_graph

class ToyData(Dataset):
    def __init__(self, points_per_sample=200, show_data=False, transform=None):
        self.toy_data(points_per_sample, show_data)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        visual_example = self.x[index]
        y_label = self.y[index]

        if self.transform:
            self.x = self.transform(self.x)

        return (visual_example, y_label)

    def toy_data(self, points_per_sample, show_data):
        """
        Makes ToyData Example From
        Shen, Yuming, et al. "Invertible zero-shot recognition flows." ECCV. Springer, Cham, 2020.
        """
        self.cs = torch.tensor([[0., 1.], [0., 0.], [1., 0.]])
        self.cu = torch.tensor([1., 1.])
        base_dist = dist.Normal(torch.zeros(2), torch.ones(2) / 3)

        self.x = []
        for c in self.cs:
            self.x.append(F.pad(input=2 * c - 1 + base_dist.sample((points_per_sample,)),
                                pad=(0, 2, 0, 0), mode="constant", value=0).reshape(-1, 4))

        self.x_unseen = F.pad(input=2 * self.cu - 1 + base_dist.sample((points_per_sample,)),
                              pad=(0, 2, 0, 0), mode="constant", value=0).reshape(-1, 4)

        if show_data:
            temp = self.x.copy()
            temp.append(self.x_unseen)
            make_toy_graph(temp, "Toy Data", "Toy Data", fit=False, show=False, save=True)

        self.x = torch.stack(self.x).reshape(-1, 4)
        self.y = torch.stack((torch.full((points_per_sample,), 0),
                              torch.full((points_per_sample,), 1),
                              torch.full((points_per_sample,), 2))).reshape(-1)

        self.y_unseen = torch.full((points_per_sample,), 3)
