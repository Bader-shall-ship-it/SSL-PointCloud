import torch
from typing import Tuple

def feature_transformation_regularizer(feature_matrix: torch.Tensor) -> torch.Tensor:
    feature_dim = feature_matrix.shape[1]
    AAT = feature_matrix.bmm(feature_matrix.permute(0, 2, 1)).to(feature_matrix.device)
    loss = AAT - torch.eye(feature_dim).to(feature_matrix.device)
    return torch.norm(loss, dim=(1,2)).mean()

def get_pairwise_sim(left: torch.Tensor, right: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    all_data = torch.vstack([left, right])
    all_norm = torch.norm(all_data, dim=1)
    
    # dot product left vs all
    num_1 = left.matmul(all_data.t())
    denom_1 = torch.outer(torch.norm(left, dim=1), all_norm.t())
    sim_1 = num_1.divide(denom_1)
    
    # Same with the right augs
    num_2 = right.matmul(all_data.t())
    denom_2 = torch.outer(torch.norm(right, dim=1), all_norm.t())
    sim_2 = num_2.divide(denom_2)

    return sim_1, sim_2

def get_frankenstein_pairwise_sim(data: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    num_1 = left.matmul(data.t())
    denom_1 = torch.outer(torch.norm(left, dim=1), (torch.norm(data.t(), dim=0)))
    sim_mat_1 = num_1.divide(denom_1)

    num_2 = right.matmul(data.t())
    denom_2 = torch.outer(torch.norm(right, dim=1), (torch.norm(data.t(), dim=0)))
    sim_mat_2 = num_2.divide(denom_2)

    pos_dot = (left * right).sum(dim=1)
    pos_dot_norms = torch.norm(left, dim=1) * torch.norm(right, dim=1)
    pos_sims = pos_dot.divide(pos_dot_norms)

    # Hack together as if we did a augmented comparison with original for each pair.
    diag_idx = torch.arange(sim_mat_1.shape[0])
    sim_mat_1[diag_idx, diag_idx] = pos_sims
    sim_mat_2[diag_idx, diag_idx] = pos_sims
    
    return sim_mat_1, sim_mat_2

def contrastive_loss(left: torch.Tensor, right: torch.Tensor, tau: float) -> torch.Tensor:
    N = left.shape[0]
    sim_1, sim_2 = get_pairwise_sim(left, right)
    
    s1 = torch.exp(sim_1 * (1/tau))    
    s2 = torch.exp(sim_2 * (1/tau))

    idx = torch.arange(N)
    mask_1 = torch.ones_like(sim_1)
    mask_1[idx, idx] = 0

    idx = torch.arange(N)
    mask_2 = torch.ones_like(sim_2)
    mask_2[idx, idx + N] = 0
    
    s1_denom = s1.masked_select(mask_1.bool()).reshape(N, -1)
    s1_denom = torch.sum(s1_denom, dim=1)
    s2_denom = s2.masked_select(mask_2.bool()).reshape(N, -1)
    s2_denom = torch.sum(s2_denom, dim=1)
    
    s1_pos = s1[idx, idx+N]
    s2_pos = s2[idx, idx]
    
    L1 = -torch.log(s1_pos.divide(s1_denom))
    L2 = -torch.log(s2_pos.divide(s2_denom))
    loss = (L1.sum() + L2.sum()) / (2*N)
    
    return loss

def frankenstein_contrastive_loss(data: torch.Tensor, left: torch.Tensor, right: torch.Tensor, tau: float) -> torch.Tensor:
    N = data.shape[0]
    sim_1, sim_2 = get_frankenstein_pairwise_sim(data, left, right)
    
    s1 = torch.exp(sim_1 * (1/tau))
    s2 = torch.exp(sim_2 * (1/tau))
    
    s1_denom = s1.sum(dim=1)
    s2_denom = s2.sum(dim=1)
    
    s1_pos = s1.diag()
    s2_pos = s2.diag()
    
    L1 = -torch.log(s1_pos.divide(s1_denom))
    L2 = -torch.log(s2_pos.divide(s2_denom))
    
    loss = (L1.sum() + L2.sum()) / (2*N)
    
    return loss