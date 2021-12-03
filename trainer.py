from typing import Optional, List, Tuple
import dataclasses
import torch
import torch.nn.functional as F
import numpy as np
from losses import feature_transformation_regularizer, frankenstein_contrastive_loss, contrastive_loss
import utils
from models import PointNet, PointNetContrastive

class ContrastiveTrainer:
    model: torch.nn.Module
    optimizer: torch.optim
    tau: float  
    device: torch.device 

    def train_epoch(self, data: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> List[float]:
        NotImplementedError


@dataclasses.dataclass
class SupervisedTrainer:
    model: torch.nn.Module
    optimizer: torch.optim
    # scheduler: Optional[torch.optim.lr_scheduler.StepLR]
    device: torch.device

    @staticmethod
    def initialize(n_classes: int, learning_rate: float, device: torch.device) -> 'SupervisedTrainer':
        model = PointNet(num_classes=n_classes, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        return SupervisedTrainer(model=model, optimizer=optimizer, device=device)

    def train_epoch(self, data, labels, batch_size):
        N = data.shape[0]
        losses = []
        device = data.device
        self.model.train()
        for i in torch.arange(0, N, batch_size):
            train_d = data[i: i+batch_size]
            train_target = labels[i: i+batch_size].type(torch.LongTensor).to(device)
            train_d = train_d.permute(0, 2, 1)

            self.optimizer.zero_grad()
            pred, feat_trans_matrix = self.model(train_d)
            loss = F.cross_entropy(pred, train_target)
            loss += feature_transformation_regularizer(feat_trans_matrix) * 0.001

            loss.backward()
            self.optimizer.step()
            losses += [loss.item()]

    
    @torch.no_grad()
    def predict(self, data, batch_size = 64) -> torch.Tensor:
        self.model.eval()
        input = data.permute(0, 2, 1)
        preds, _ = self.model(input)
        return preds
    
    @torch.no_grad()
    def evaluate(self, data, labels, batch_size = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        N = data.shape[0]
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        for i in torch.arange(0, N, batch_size):
            input = data[i: i+batch_size].to(self.device)
            input = input.permute(0, 2, 1)
            target = labels[i: i+batch_size].type(torch.LongTensor).to(self.device)

            pred, feat_trans_matrix = self.model(input)
            loss = F.cross_entropy(pred, target)
            loss += feature_transformation_regularizer(feat_trans_matrix) * 0.001
            total_loss += loss
            acc = utils.get_accuracy(pred, target)
            total_acc+= acc
        
        batches = np.ceil(N/batch_size)
        return loss/batches, total_acc/batches


@dataclasses.dataclass
class SimCLRTrainer(ContrastiveTrainer):
    model: torch.nn.Module
    optimizer: torch.optim
    tau: float  
    device: torch.device 

    @staticmethod
    def initialize(learning_rate: float, temperature: float, device: torch.device) -> 'SimCLRTrainer':
        model = PointNetContrastive(device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        return SimCLRTrainer(model=model, optimizer=optimizer, tau=temperature, device=device)
        
    def train_epoch(self, data: torch.Tensor, left: torch.Tensor, right: torch.Tensor, batch_size: int) -> List[float]:
        N = left.shape[0]
        losses = []
        self.model.train()
        for i in torch.arange(0, N, batch_size):
            # (BATCH, n_points, C) -> (BATCH, C, n_points)
            left_batch = left[i: i+batch_size]
            left_batch = left_batch.permute(0, 2, 1).to(self.device)
            right_batch = right[i: i+batch_size]
            right_batch = right_batch.permute(0, 2, 1).to(self.device)

            self.optimizer.zero_grad()
            # Pass our data through the feature transform h(.) then into projection space g(.)
            left_h, _ = self.model(left_batch)
            right_h, _ = self.model(right_batch)
            loss = contrastive_loss(left_h, right_h, tau=self.tau)
            loss.backward()
            self.optimizer.step()
            # losses += [loss.item()]

        return losses

    

@dataclasses.dataclass
class ModifiedSimCLRTrainer(ContrastiveTrainer):
    model: torch.nn.Module
    optimizer: torch.optim
    tau: float  
    device: torch.device

    @staticmethod
    def initialize(learning_rate: float, temperature: float, device: torch.device) -> 'ModifiedSimCLRTrainer':
        model = PointNetContrastive(device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        return ModifiedSimCLRTrainer(model=model, optimizer=optimizer, tau=temperature, device=device)

    def train_epoch(self, data: torch.Tensor, left: torch.Tensor, right: torch.Tensor, batch_size: int) -> List[float]:
        N = data.shape[0]
        self.model.train()
        losses = []
        for i in torch.arange(0, N, batch_size):
            
            data_batch = data[i: i+batch_size].permute(0, 2, 1)
            left_batch = left[i: i+batch_size].permute(0, 2, 1)
            right_batch = right[i: i+batch_size].permute(0, 2, 1)

            self.optimizer.zero_grad()
            data_h, _ = self.model(data_batch)
            left_h, _ = self.model(left_batch)
            right_h, _ = self.model(right_batch)

            loss = frankenstein_contrastive_loss(data_h, left_h, right_h, self.tau)
            loss.backward()
            self.optimizer.step()
            # losses += [loss.item()]

        return losses