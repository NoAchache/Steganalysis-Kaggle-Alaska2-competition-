import os
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.optim import lr_scheduler

from utils import cfg, LossTracker, ModelUtils, TensorboardLogging, CompetitionMetric
from data_provider import TrainDataLoader, BaseTransform, AugmentatedTransform
from models import EndToEndNet, Criterion, AdamW
from models.network import SrmFiltersSetter


class Train:
    def __init__(self):
        self.train_step = 0
        self.epoch = 0
        self.lr = cfg.lr
        self.model = EndToEndNet()
        self.criterion = Criterion()
        lrs = [{"params": self.model.srm.parameters(), "lr": cfg.lrs[0]},
               {"params": self.model.efficient_net.parameters(), "lr": cfg.lrs[1]},
               ]

        self.optimizer = AdamW(lrs)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, **cfg.lr_scheduler_params)
        logs_dir = os.path.join(cfg.log_dir, cfg.train_name)
        self.writer = TensorboardLogging(logs_dir)

    def main(self):
        if cfg.device.type == 'cuda':
            if cfg.enable_lms:
                torch.cuda.set_enabled_lms(True)
            cudnn.benchmark = True

        # Loads a model only if there is an existing file in cfg.save_path
        ModelUtils.load_model(self)

        if self.train_step == 0:
            SrmFiltersSetter.initialize_filters(self.model)
            print('SRM High Pass filters initialized')

        self.model = self.model.to(cfg.device)

        train_data = TrainDataLoader(
            img_root=os.path.join(cfg.data_path, 'train'),
            transform=AugmentatedTransform(sizes=cfg.input_size),
            is_training=True
        )
        train_loader = data.DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True,
                                       num_workers=cfg.num_workers)

        val_data = TrainDataLoader(
            img_root=os.path.join(cfg.data_path, 'train'),
            transform=BaseTransform(sizes=cfg.input_size),
            is_training=False
        )
        val_loader = data.DataLoader(val_data, batch_size=cfg.batch_size, shuffle=False,
                                     num_workers=cfg.num_workers)

        print(f'Training on {len(train_data)} images')
        print('Start training.')

        for e in range(self.epoch, cfg.max_epoch):
            self._train(train_loader)
            self.epoch += 1
            if self.epoch % cfg.val_freq == 0:
                self._validation(val_loader)


    def _train(self, train_loader):
        # Show and log loss results every 100 steps
        loss_tracker = LossTracker()
        pred_scores = []
        true_scores = []

        self.model.train()

        print('Epoch: {} : LR = {}'.format(self.epoch, self.lr))

        for i, img in enumerate(train_loader):

            self.train_step += 1

            # Split the batch in 4 sub-batches, such that each sub-batch contains either the cover, JUNIWARD, JMiPOD or
            # UERD version of each image.

            batch_splits, labels = ModelUtils.batch_splitter(img, num_imgs=4)

            classification_out = [self.model(sub_batch) for sub_batch in batch_splits]
            losses = [self.criterion(c, l) for c, l in zip(classification_out, labels)]
            total_loss = sum(losses)/len(losses)

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            for batch_labs, batch_res in zip(labels, classification_out):
                ModelUtils.append_results(batch_labs, batch_res, pred_scores, true_scores)

            loss_tracker.increment_loss(total_loss)

            if i % cfg.log_freq == 0 and i > 0:
                weighted_AUC = CompetitionMetric.alaska_weighted_auc(true_scores, pred_scores)
                pred_scores = []
                true_scores = []

                loss_dict = loss_tracker.write_dict(weighted_AUC)
                loss_tracker.print_losses(self.epoch, i, len(train_loader), weighted_AUC)
                loss_tracker = LossTracker()  # Reinitialize the loss tracking
                self.writer.write_scalars(loss_dict, tag='train', n_iter=self.train_step)

            if i % cfg.save_freq == 0 and i > 0:
                ModelUtils.save_model(self)


    def _validation(self, valid_loader):
        with torch.no_grad():

            self.model.eval()
            loss_tracker = LossTracker()
            pred_scores = []
            true_scores = []

            for i, (img, label) in enumerate(valid_loader):

                img = torch.cat(img)
                img = img.to(cfg.device)
                label = torch.cat(label)
                label = label.to(cfg.device)

                classification_output = self.model(img)
                loss = self.criterion(classification_output, label)

                ModelUtils.append_results(label, classification_output, pred_scores, true_scores)
                loss_tracker.increment_loss(loss)

                if i % 100 == 0:
                    weighted_AUC = CompetitionMetric.alaska_weighted_auc(true_scores, pred_scores)
                    loss_tracker.print_losses(self.epoch, i, len(valid_loader), weighted_AUC)

            weighted_AUC = CompetitionMetric.alaska_weighted_auc(true_scores, pred_scores)
            loss_dict = loss_tracker.write_dict(weighted_AUC)
            loss_tracker.print_losses(self.epoch, i, len(valid_loader), weighted_AUC)
            self.writer.write_scalars(loss_dict, tag='val', n_iter=self.train_step)
            self.scheduler.step(metrics=loss_tracker.loss.avg)
            lr = self.optimizer.param_groups[-1]['lr']
            self.writer.write_scalars({'lr': lr}, tag='val', n_iter=self.train_step)


if __name__ == "__main__":
    trainer = Train()
    trainer.main()
