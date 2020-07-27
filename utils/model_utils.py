import os
import torch
import random

from utils import cfg

class ModelUtils:

    @staticmethod
    def save_model(model):
        save_path = os.path.join(cfg.save_path, '{}_{}_{}.pth'.format(cfg.train_name, model.epoch, model.train_step))
        print('Saving to {}.'.format(save_path))
        state_dict = {
            'lr': model.lr,
            'epoch': model.train_step,
            'model': model.model.state_dict(),
            'optimizer': model.optimizer.state_dict()
        }
        torch.save(state_dict, save_path)

    @staticmethod
    def load_model(model):
        if not os.path.exists(cfg.save_path):
            os.makedirs(cfg.save_path)

        files = os.listdir(cfg.save_path)
        if len(files) == 0:
            print('Training with a new model')
            return

        ite_list = []
        epoch_list = []

        # Load the file with the largest number of iterations
        for file in files:
            file = file.split('_')
            epoch_list.append(int(file[1]))
            ite_list.append(int(file[2][:-4]))
        max_it = max(ite_list)
        model.train_step = max_it
        model.epoch = epoch_list[ite_list.index(max_it)]
        ckpt_file = files[ite_list.index(max_it)]
        model_path = os.path.join(cfg.save_path, ckpt_file)

        print('Loading from {}'.format(model_path))
        state_dict = torch.load(model_path, map_location=cfg.device)
        model.model.load_state_dict(state_dict['model'])

    @staticmethod
    def to_device(*tensors):
        if len(tensors) < 2:
            return tensors[0].to(cfg.device)
        return tuple(t.to(cfg.device) for t in tensors)

    @staticmethod
    def batch_splitter(img, num_imgs):
        '''
        Split the batch in 4 sub-batches, such that each sub-batch contains either the cover, JUNIWARD, JMiPOD or
        UERD version of each image. This allows to infer each version separately as otherwise, it leads to problems
        with the batch_norm.
        '''

        batch_splits = [[] for _ in range(num_imgs)]
        labels = [[] for _ in range(num_imgs)]
        kind_idxs = list(range(num_imgs))

        for j in range(cfg.batch_size):
            random.shuffle(kind_idxs)
            for k in range(num_imgs):
                idx = kind_idxs[k]
                batch_splits[k].append(img[idx][j:j + 1])
                labels[k].append(idx)
        batch_splits = [torch.cat(b).to(cfg.device) for b in batch_splits]
        labels = [torch.tensor(l).to(cfg.device) for l in labels]

        return batch_splits, labels

    @staticmethod
    def append_results(batch_labels, batch_output, pred_scores, true_scores):
        for label, result in zip(batch_labels, batch_output):
            cover_score = 1 - torch.exp(result[0]).item()  # likehood of being a cover img
            pred_scores.append(cover_score)
            truth = label.cpu().numpy().clip(min=0, max=1)
            true_scores.append(truth)
