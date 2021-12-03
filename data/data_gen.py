from typing import Tuple, Optional
import torch
from .data_utils import ShapeTransformOptions
from . import data_utils

def generate_data(samples: int, shapes: int, transform_opts: Optional[ShapeTransformOptions] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    if transform_opts is None:
        transform_opts = ShapeTransformOptions(dim = 1_000)

    cube, cube_labels = data_utils.gen_cube(samples)
    cube = data_utils.transform_data(cube, transform_opts)

    cyl, cyl_labels = data_utils.gen_cylinder(samples)
    cyl = data_utils.transform_data(cyl, transform_opts)

    cone, cone_labels = data_utils.gen_cone(samples)
    cone = data_utils.transform_data(cone, transform_opts)

    tor, tor_labels = data_utils.gen_torus(samples)
    transform_opts.noise_std = min(0.03, transform_opts.noise_std)
    tor = data_utils.transform_data(tor, transform_opts)
    
    if shapes < 4:
        data = torch.vstack([cube, cyl, cone])
        labels = torch.hstack([cube_labels, cyl_labels, cone_labels])
    else:
        data = torch.vstack([cube, cyl, cone, tor])
        labels = torch.hstack([cube_labels, cyl_labels, cone_labels, tor_labels])
    
    return data, labels

def augment(samples: torch.Tensor, min_shear: float = -3, max_shear: float = 3) -> torch.Tensor:
    '''Shear matrix as an augmentation step.'''
    batch_size = samples.shape[0]
    sh = torch.empty(batch_size, 3, 3).uniform_(min_shear, max_shear)
    iden = torch.eye(3)
    mask = torch.ones((3,3)) - iden
    shear_matrix = (sh * mask) + iden
    shear_matrix = shear_matrix.to(samples.device)
    return samples.bmm(shear_matrix)

def normalize_pointcloud(data: torch.Tensor) -> torch.Tensor:
    # assert data.dim == 3
    means = data.mean(dim=(1)).unsqueeze(dim=1)
    stdevs = data.std(dim=(1)).unsqueeze(dim=1)
    return data - means / stdevs

def generate_pairs(data, augment_1=augment, augment_2=augment, normalize: bool = False, device: torch.device = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    left = augment_1(data).to(device)
    right = augment_2(data).to(device)

    if normalize:
        left = normalize_pointcloud(left)
        right = normalize_pointcloud(right)
    
    return left, right
