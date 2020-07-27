# from easydict import EasyDict
import torch
import os.path as osp

PROJECT_ROOT = ''

class Object(object):
    pass

cfg = Object()

cfg.train_name = 'test'

cfg.project_root = PROJECT_ROOT
cfg.data_path = osp.join(PROJECT_ROOT, 'data')
cfg.save_path = osp.join(PROJECT_ROOT, 'save')  # Path to save and load checkpoint models
cfg.log_dir = osp.join(PROJECT_ROOT, 'log')

cfg.train_data_share = 0.8 # Share of the dataset used for training
cfg.input_size = [512, 512]
cfg.max_epoch = 50
cfg.val_freq = 1  # run validation every # epochs
cfg.save_freq = 1000 # save checkpoint every # iterations
cfg.log_freq = 500
cfg.batch_size = 4
cfg.num_workers = 6

cfg.lr_decay_rate = 10000 # decay lr every # iterations
cfg.lr = 1e-3
cfg.lr_scheduler_params = {'factor': 0.5, 'patience': 1, 'threshold_mode': 'abs'}


# cfg.device = torch.device('cpu')
cfg.device = torch.device('cuda')
cfg.enable_lms = True # lms is module developed by ibm which allows to train with smaller gpus
cfg.lrs = [0.00001, 0.001]


