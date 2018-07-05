import sys
import logging
from datetime import datetime
import pprint

sys.path.extend(['..'])

import tensorflow as tf

from dataloaders import DatasetLoader, DatasetFileLoader

from models import LeNet, ResNet18, ResNet50, AlexNet, Inception, ResNeXt
from models import ResNet50_MI

from trainers.MTrainer import MTrainer

from utils.logger import DefinedSummarizer
from utils.utils import get_args
from utils.dirs import create_dirs

from config import Config


def main():
    # create the experiments dirs
    create_dirs([Config.summary_dir, Config.checkpoint_dir, "logs"])

    handlers = [logging.FileHandler(datetime.now().strftime(f"./logs/%Y-%m-%d_%H-%M-%S-Log.log")),
                logging.StreamHandler()]

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, handlers=handlers)

    logging.info("Started Logging")
    logging.info(f"Number of cores: {pprint.pformat(Config.num_parallel_cores)}")
    logging.info(f"Address of GPU used for training: {pprint.pformat(Config.gpu_address)}")

    logging.info(f"Type of DataLoader: {pprint.pformat(Config.dataloader_type)}")

    logging.info(f"Type of Model: {pprint.pformat(Config.model_type)}")
    logging.info(f"Number of Epochs: {pprint.pformat(Config.num_epochs)}")
    logging.info(f"Optimizer Type: {pprint.pformat(Config.optimizer_type)}")
    logging.info(f"Optimizer parameters: {pprint.pformat(Config.optim_params)}")
    logging.info(f"Scheduler Type: {pprint.pformat(Config.lr_scheduler_type)}")
    logging.info(f"Scheduler parameters: {pprint.pformat(Config.lr_scheduler_params)}")
    logging.info(f"Train/Validation split ratio: {pprint.pformat(Config.train_val_split)}")
    logging.info(f"Batch size: {pprint.pformat(Config.batch_size)}")

    logging.info(f"Training on Subset of the data: {pprint.pformat(Config.train_on_subset)}")
    logging.info(f"Training on Subset of size: {pprint.pformat(Config.subset_size)}")

    logging.info(f"Generating Patches: {pprint.pformat(Config.train_on_patches)}")
    logging.info(f"Patch size (square): {pprint.pformat(Config.patch_size)}")

    # create your data generator on the CPU 

    with tf.device("/cpu:0"):

        if (Config.dataloader_type.lower() == 'datasetfileloader'):
            data_loader = DatasetFileLoader.DatasetFileLoader(Config)
        else:
            data_loader = DatasetLoader.DatasetLoader(Config)

    # create tensorflow session on the GPU defined in Config file

    with tf.device(Config.gpu_address):
    
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

            # create instance of the model you want
            if (Config.model_type.lower() == 'lenet'):
                model = LeNet.LeNet(data_loader, Config)
            elif (Config.model_type.lower() == 'resnet18'):
                model = ResNet18.ResNet18(data_loader, Config)
            elif (Config.model_type.lower() == 'resnet50'):
                model = ResNet50_MI.ResNet50_MI(data_loader, Config)
            elif (Config.model_type.lower() == 'alexnet'):
                model = AlexNet.AlexNet(data_loader, Config)
            elif (Config.model_type.lower() == 'inception'):
                model = Inception.Inception(data_loader, Config)
            elif (Config.model_type.lower() == 'resnext'):
                model = ResNeXt.ResNeXt(data_loader, Config)
            else:
                model = LeNet.LeNet(data_loader, Config)

            # create tensorboard logger
            logger = DefinedSummarizer(sess, summary_dir=Config.summary_dir,
                                       scalar_tags=['train/loss_per_epoch', 'train/acc_per_epoch',
                                                    'test/loss_per_epoch', 'test/acc_per_epoch', 'learning_rate', 'si_weight', 'mi_weight'])

            # create trainer and path all previous components to it
            trainer = MTrainer(sess, model, Config, logger, data_loader)

            # here you train your model
            trainer.train()


if __name__ == '__main__':
    main()