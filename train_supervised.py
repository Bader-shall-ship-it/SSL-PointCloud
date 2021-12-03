import argparse
import torch
from data import data_gen
from data.data_utils import train_validation_split, ShapeTransformOptions
from trainer import SupervisedTrainer
import utils
from train import train_supervised

def main(args: argparse.Namespace):
    device = args.device
    examples_per_shape = args.num_samples_per_shape
    sample_dim = args.shape_dim
    val_split = args.val_split
    lr = args.lr
    training_epochs = args.epochs
    batch_size = args.batch_size

    full_data, full_labels = data_gen.generate_data(examples_per_shape, 3, transform_opts=ShapeTransformOptions(dim=sample_dim))
    train_data, train_labels, test_data, test_labels = train_validation_split(full_data, full_labels, val_percent=val_split)
    k = torch.unique(full_labels).shape[0]
    print("Training samples: {}, Test samples: {}, classes: {}".format(train_data.shape[0], test_data.shape[0], k))

    trainer = SupervisedTrainer.initialize(k, lr, device)
    checkpoint_dir = "./checkpoints/"+utils.config_str("supervised", k, examples_per_shape, sample_dim=sample_dim, include_ts=True)
    train_supervised(trainer, train_data, train_labels, training_epochs, batch_size=batch_size, save_path=checkpoint_dir, save_name="cls_model_")

    print(f"[+] Finished training procedure, with checkout path {checkpoint_dir}")
    loss, acc = trainer.evaluate(test_data, test_labels)
    print("TEST ACCURACY: {}".format(acc.item()))

    cf = utils.confusion_matrix(trainer.predict(test_data), test_labels, k, normalize=False)
    utils.visualize_confusion_matrix(cf, k, ['Cube', 'Cyl', 'Cone', 'Torus'], save_path = "./checkpoints/"+checkpoint_dir+'confusion_matrix.png')


if __name__ == '__main__':
    args = utils.parse_arg()
    main(args)
