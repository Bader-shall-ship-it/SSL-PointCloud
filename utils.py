import torch
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_weights", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_samples_per_shape", type=int, default=2_500,
                    help="number of samples to generate per shape (default: %(default)s)")
    parser.add_argument("--shape_dim", type=int, default=1_000)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--device", type=available_device, default='cuda')
    parser.add_argument("--loss", type=str, default='modifiedSimCLR', choices=['SimCLR', 'modifiedSimCLR'],
                    help='type of contrastive loss to use (default: %(default)s)')
    parser.add_argument("--aug_normalize", type=bool, default=False,
                    help="use point cloud normalization in augmentation (default: %(default)s)")
    args = parser.parse_args()
    return args

def available_device(device: str) -> torch.device:
    return device if torch.cuda.is_available() else 'cpu'

def save_model(model: torch.nn.Module, epoch: int, lr: float, path, name) -> None:
    state = prepare_checkpoint(model, epoch, lr)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    torch.save(state, path+name+'%d.pth' % (epoch))

def prepare_checkpoint(model: torch.nn.Module, epoch: int, lr: float):
    chkp = {}
    chkp["model"] = model.state_dict()
    # chkp["curr_epoch"] = epoch
    # chkp["lr"] = lr
    return chkp

def get_accuracy(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    p = torch.argmax(pred, dim=1)
    return (p == gt).sum() / gt.shape[0]

def config_str(model_type: str, k: int, num_samples: int, sample_dim: int, include_ts: bool = True) -> str:
    ts = ""
    if include_ts:
        from datetime import datetime
        ts = "_{:%d-%m-%Y_%HH-%MM-%Ss}".format(datetime.now())
    return model_type + "/{}k_{}n_{}dim{}/".format(k, num_samples, sample_dim, ts)


def confusion_matrix(preds: torch.Tensor, gt: torch.Tensor, classes: int, normalize: bool = True) -> torch.Tensor:
    accum = torch.zeros((classes, classes))
    nk = torch.arange(classes)

    if preds.dim() > 1:
        preds = torch.argmax(preds, dim=1)
        
    for k in nk:
        idx = gt == k
        results = preds[idx]
        for r in nk:
            accum[k, r] += (results == r).sum()
    
    if normalize:
        accum = accum.divide(accum.sum(dim=1).reshape(-1, 1))
    return accum

def visualize_confusion_matrix(matrix: torch.Tensor, classes: int, labels: List[str] = ["Cube", "Cyl", "Cone", "Torus"], cmap: str = 'plasma', save_path: str = './confusion_matrix.pdf'):
    fig, ax = plt.subplots()
    ax.matshow(matrix.numpy(), cmap=cmap, interpolation='nearest')

    ax.set_xticks(np.arange(classes))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(classes))
    ax.set_yticklabels(labels)

    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position('top')

    for (y, x), value in np.ndenumerate(matrix.numpy()):
        color = "#FFFFFF" if value < 0.6 else "#000000"
        ax.text(x, y, f"{value*100:.1f}%", va="center", ha="center", color=color)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')

    