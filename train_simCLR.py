import argparse
import torch
from data import data_gen
from data.data_utils import train_validation_split, ShapeTransformOptions
import utils
from train import train_simclr
from trainer import SimCLRTrainer, ModifiedSimCLRTrainer

def main(args: argparse.Namespace):
    device = args.device
    training_epochs = args.epochs
    examples_per_shape = args.num_samples_per_shape
    sample_dim = args.shape_dim
    val_split = args.val_split
    lr = args.lr
    use_normalization_as_aug = args.aug_normalize
    temperature = 0.3

    full_data, full_labels = data_gen.generate_data(examples_per_shape, 3, transform_opts=ShapeTransformOptions(dim=sample_dim))
    train_data, train_labels, test_data, test_labels = train_validation_split(full_data, full_labels, val_percent=val_split)
    k = torch.unique(full_labels).shape[0]
    print("Training samples: {}, Test samples: {}".format(train_data.shape[0], test_data.shape[0]))
    
    # simCLR
    data = train_data
    left, right = data_gen.generate_pairs(data, normalize=use_normalization_as_aug, device=device)

    checkpoint_dir = "./checkpoints/"+utils.config_str("simCLR", k, examples_per_shape, sample_dim=sample_dim, include_ts=True)
    if args.loss == 'SimCLR':
        trainer = SimCLRTrainer.initialize(lr, temperature, device)
    elif args.loss == 'modifiedSimCLR':
        trainer = ModifiedSimCLRTrainer.initialize(lr, temperature, device)

    train_simclr(trainer, data, left, right, epochs=training_epochs, batch_size=args.batch_size, scheduler=None, save_path=checkpoint_dir, save_name='simCLR_model_')

    print(f"[+] Finished training, with checkout path {checkpoint_dir}")
    

if __name__ == '__main__':
    args = utils.parse_arg()
    main(args)