import numpy as np
import torch
import pytorch3d
import pytorch3d.transforms
from typing import Tuple
import dataclasses

# TODO: This should be an IntEnum
CUBE_LABEL = 0
CYLINDER_LABEL = 1
CONE_LABEL = 2
TORUS_LABEL = 3
SPHERE_LABEL = 4

@dataclasses.dataclass
class ShapeTransformOptions:
    dim: int
    noise_std: float = 0.05
    scale_min: float = 1
    scale_max: float = 5
    translation: float = 0.8

    def copy(self):
        return dataclasses.replace(self)


def gen_cube(n_samples: int, n_points_h: int = 8, r: float = 0.5, label: str = CUBE_LABEL) -> Tuple[torch.Tensor, torch.Tensor]:

    X, Y = np.mgrid[-r:r:r/n_points_h, -r:r:r/n_points_h]
    X, Y = X.flatten(), Y.flatten()
    c = np.ones(X.shape[0]) * r
    points = np.hstack((np.vstack((c, X, Y)),
                       np.vstack((-1*c, X, Y)),
                       np.vstack((Y, c, X)),
                       np.vstack((Y, -1*c, X)),
                       np.vstack((X,Y, c)),
                       np.vstack((X, Y, -1*c))
                     )).T
    accum = np.tile(points, (n_samples, 1))
    
    labels = torch.ones(n_samples) * label
    return torch.Tensor(accum).reshape(n_samples, -1, 3), labels


def gen_cylinder(n_samples: int, n_points_h: int = 30, n_points_c: int = 50, r: float=0.5, label: str = CYLINDER_LABEL) -> Tuple[torch.Tensor, torch.Tensor]:
    deg2rad = 3.14/180
    angles = torch.arange(0, 360, 360/n_points_c).reshape(-1, 1)
    
    x = torch.cos(angles * deg2rad) * r
    x = torch.tile(x, (n_points_h, 1))
    z = torch.sin(angles * deg2rad) * r
    z = torch.tile(z, (n_points_h, 1))
    y = torch.arange(-r, r, 2*r/n_points_h)
    y = torch.repeat_interleave(y, n_points_c).reshape(-1, 1)
    
    cyl = torch.stack([x, y, z], dim=1).squeeze()
    cyls = cyl.repeat((n_samples, 1, 1))
    labels = torch.ones(n_samples) * label
    return cyls, labels


def gen_cone(n_samples: int, n_points_h: int = 30, n_points_c: int = 50, r: float=0.5, label: str = CONE_LABEL) -> Tuple[torch.Tensor, torch.Tensor]:
    deg2rad = 3.14/180
    # TODO: rand slope per n_sample.
    cone_slope = 0.75
    
    y = torch.arange(-r, r, 2*r/n_points_h)
    y = torch.repeat_interleave(y, n_points_c).reshape(-1, 1)
    angles = torch.arange(0, 360, 360/n_points_c).reshape(-1, 1)
    x = torch.cos(angles * deg2rad) * r
    x = torch.tile(x, (n_points_h, 1))
    x = (0.5 - y) * cone_slope * x
    z = torch.sin(angles * deg2rad) * r
    z = torch.tile(z, (n_points_h, 1))
    z = (0.5 - y) * cone_slope * z
    
    cone = torch.stack([x, y, z], dim=1).squeeze()
    cones = cone.repeat((n_samples, 1, 1))
    labels = torch.ones(n_samples) * label
    return cones, labels

def gen_torus(n_samples: int, n_points_h: int = 30, n_points_c: int = 50, inner: float = 0.25, outer: float = 0.5, label: str = TORUS_LABEL) -> Tuple[torch.Tensor, torch.Tensor]:
    deg2rad = 3.14/180
    thetas = torch.arange(0, 360, 360/n_points_c).reshape(-1, 1)
    thetas = torch.tile(thetas, (n_points_h, 1))
    phis = torch.arange(0, 360, 360/n_points_h).reshape(-1, 1)
    phis = torch.repeat_interleave(phis, n_points_c).reshape(-1, 1)
    x = (outer + inner*torch.cos(thetas * deg2rad)) * torch.cos(phis * deg2rad)
    y = (outer + inner*torch.cos(thetas * deg2rad)) * torch.sin(phis * deg2rad)
    z = inner*torch.sin(thetas * deg2rad)
    tor =  torch.stack([x, y, z], dim=1).squeeze()
    labels = torch.ones(n_samples) * label
    return tor.repeat((n_samples, 1, 1)), labels

def gen_sphere(n_samples=1, lat=50, lon=50, r=0.5, label=SPHERE_LABEL) -> Tuple[torch.Tensor, torch.Tensor]:
    '''Not used'''  
    lat_stride = 360 / lat
    lon_stride = 360 / lon

    deg2rad = 3.14/180
    points = [(r*np.sin(ln*deg2rad),
               r*np.cos(ln*deg2rad)*np.cos(lt*deg2rad),
               r*np.cos(ln*deg2rad)*np.sin(lt*deg2rad)) 
            for lt in np.arange(0, 360, lat_stride)
            for ln in np.arange(0, 360, lon_stride)
            for _ in np.arange(n_samples)]
    
    labels = torch.ones(n_samples) * label
    return torch.Tensor(points).reshape(n_samples, -1, 3), labels

def transform_data(data: torch.Tensor, opt: ShapeTransformOptions) -> torch.Tensor:
    assert data.dim() == 3 
    assert data.shape[2] == 3

    samples_per_shape = opt.dim
    min_scale, max_scale = opt.scale_min, opt.scale_max
    translation_limit = opt.translation
    noise_std = opt.noise_std

    idx = torch.randperm(data.shape[1])[:samples_per_shape]
    data = data[:, idx, :]
    batch_size = data.shape[0]
    feature_dim = data.shape[1]
    
    rot = pytorch3d.transforms.random_rotations(batch_size) # rotation
    scale = pytorch3d.transforms.Scale(torch.FloatTensor(batch_size).uniform_(min_scale, max_scale)).get_matrix()[:, :3, :3] # uniform scaling
    M = rot.bmm(scale)
    trans = torch.empty(batch_size, 3).uniform_(-translation_limit, translation_limit) # translation
    trans = trans.repeat_interleave(feature_dim, dim=0).reshape(batch_size, feature_dim, 3)
    noise = torch.empty(*data.shape).uniform_(-noise_std, noise_std)
    return (data + noise).bmm(M) + trans


def train_validation_split(data: torch.Tensor, labels: torch.Tensor, val_percent: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    N = data.shape[0]
    idx = torch.randperm(N)
    train_idx = int((1-val_percent)*N)
    train_data = data[idx[:train_idx]]
    train_labels = labels[idx[:train_idx]]
    test_data = data[idx[train_idx:]]
    test_labels = labels[idx[train_idx:]]
    return train_data, train_labels, test_data, test_labels