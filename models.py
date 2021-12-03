from typing import Tuple, List
import torch
import torch.nn.functional as F
import einops

class PointwiseMax(torch.nn.Module):
    """A maxpool layer that produces dim-wise pooling. E.g. for a batch of shapes (B, N, 3), we want (B, 3)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.reduce(x, 'b c d -> b c', 'max')

class PointNetContrastive(torch.nn.Module):
    """The PointNet feature backbone with a contrastive projection head g(.)."""

    def __init__(self, feature_dim: int = 128, device: torch.device = "cpu"):
        super().__init__()
        self.f = PointNetFeature(device=device)
        self.g = torch.nn.Sequential(torch.nn.Linear(1024, 512, bias=False), 
                                    torch.nn.BatchNorm1d(512),
                                    torch.nn.ReLU(inplace=True), 
                                    torch.nn.Linear(512, feature_dim)).to(device)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, trans_feat_mat = self.f(input)
        out = self.g(x)
        return out, trans_feat_mat


class PointNet(torch.nn.Module):
    """The vanilla PointNet implementation."""

    def __init__(self, num_classes: int, device: torch.device = "cpu"):
        super().__init__()
        self.feature = PointNetFeature(device=device)
        
        input_channel = 1024
        self.fc_layer = torch.nn.Sequential(torch.nn.Linear(input_channel, 512, bias=False),
                                            torch.nn.BatchNorm1d(512),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(512, 256, bias=False),
                                            torch.nn.Dropout(0.3),
                                            torch.nn.BatchNorm1d(256),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(256, num_classes)).to(device)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, trans_feat_mat = self.feature(input)
        logits = self.fc_layer(x)
        return logits, trans_feat_mat

class PointNetFeature(torch.nn.Module):
    """PointNet feature backbone network."""

    def __init__(self, mlp1: List[int] = [3, 64, 64], mlp2: List[int] = [64, 64, 128, 1024], device: torch.device = "cpu"):
        super().__init__()
        self.input_transform = Tnet(k=3, device=device)
        
        first_mlp = torch.nn.ModuleList()
        input_channel = mlp1[0]
        for out_channel in mlp1[1:]:
            first_mlp.append(torch.nn.Conv1d(input_channel, out_channel, 1))
            first_mlp.append(torch.nn.BatchNorm1d(out_channel))
            first_mlp.append(torch.nn.ReLU())
            input_channel = out_channel
        
        self.first_mlp = torch.nn.Sequential(*first_mlp).to(device)
        self.feature_transform = Tnet(k=64, device=device)

        second_mlp = torch.nn.ModuleList()
        for i in torch.arange(len(mlp2)-2):
            second_mlp.append(torch.nn.Conv1d(mlp2[i], mlp2[i+1], 1))
            second_mlp.append(torch.nn.BatchNorm1d(mlp2[i+1]))
            second_mlp.append(torch.nn.ReLU())
        second_mlp.append(torch.nn.Conv1d(mlp2[-2], mlp2[-1], 1))
        second_mlp.append(torch.nn.BatchNorm1d(mlp2[-1]))
        self.second_mlp = torch.nn.Sequential(*second_mlp).to(device)


    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ts = self.input_transform(input)
        x = input.permute(0, 2, 1).bmm(ts).permute(0, 2, 1)
        x = self.first_mlp(x)

        ts2 = self.feature_transform(x)
        x = x.permute(0, 2, 1).bmm(ts2).permute(0, 2, 1)
        x = self.second_mlp(x)

        x = einops.reduce(x, 'b c d -> b c', 'max')
        return x, ts2.to(x.device)

class Tnet(torch.nn.Module):
    """PointNet transformation network. k=3 for input transformation, k=64 for feature transformation"""

    def __init__(self, k: int = 3, conv_layers: List[int] = [64, 128, 1024], device: torch.device = "cpu"):
        super().__init__()
        self.k = k
        self.model = torch.nn.Sequential(torch.nn.Conv1d(k, conv_layers[0], 1),
                                        torch.nn.BatchNorm1d(conv_layers[0]),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv1d(64, conv_layers[1], 1),
                                        torch.nn.BatchNorm1d(conv_layers[1]),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv1d(128, conv_layers[2], 1),
                                        torch.nn.BatchNorm1d(conv_layers[2]),
                                        PointwiseMax(),
                                        torch.nn.Linear(conv_layers[2], 512, bias=False),
                                        torch.nn.BatchNorm1d(512),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(512, 256, bias=False),
                                        torch.nn.BatchNorm1d(256),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(256, k*k)).to(device)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.model(input)
        x = x.reshape(-1, self.k, self.k)
        # NOTE(Bader) "initialize identity output".
        output = torch.eye(self.k).to(x.device) + x
        return output


