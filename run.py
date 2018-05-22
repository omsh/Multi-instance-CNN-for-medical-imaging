import sys
import logging
from datetime import datetime
import pprint

sys.path.extend(['..'])

import tensorflow as tf

from Loader import DatasetLoader
from models.MModel import MModel
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
    
    logging.info(f"Number of Epochs: {pprint.pformat(Config.num_epochs)}")
    logging.info(f"Learning Rate: {pprint.pformat(Config.learning_rate)}")
    logging.info(f"Train/Validation split ratio: {pprint.pformat(Config.train_val_split)}")
    logging.info(f"Batch size: {pprint.pformat(Config.batch_size)}")
    
    logging.info(f"Training on Subset of the data: {pprint.pformat(Config.train_on_subset)}")
    logging.info(f"Training on Subset of size: {pprint.pformat(Config.subset_size)}")

    # create tensorflow session
    sess = tf.Session()

    # create your data generator
    data_loader = DatasetLoader(Config)

    # create instance of the model you want
    model = MModel(data_loader, Config)

    # create tensorboard logger
    logger = DefinedSummarizer(sess, summary_dir = Config.summary_dir, 
                               scalar_tags=['train/loss_per_epoch', 'train/acc_per_epoch',
                                            'test/loss_per_epoch','test/acc_per_epoch'])

    # create trainer and path all previous components to it
    trainer = MTrainer(sess, model, Config, logger, data_loader)

    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()