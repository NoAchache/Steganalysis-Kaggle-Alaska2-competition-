from time import time


class LossTracker(object):
    def __init__(self):
        self.initiate_loss_tracking()
        self.counter = 0

    def initiate_loss_tracking(self):
        self.loss = AverageMeter()

        self.timer = AverageMeter()
        self.start_time = time()

    def _update_loss(self, attribute, loss):
        if loss.item() > 0:
            attribute.update(loss.item())

    def increment_loss(self, loss):
        self._update_loss(self.loss, loss)

        self.timer.update(time() - self.start_time)
        self.start_time = time()

        self.counter += 1

    def print_losses(self, epoch, ite, len_loader, weighted_AUC):
        print(
            'epoch: {} ({:d} / {:d}). Avg over the last {:d} steps. {:.2f} s/step. - Loss: {:.4f}, '
            'Weighted AUC: {:.2f}%'.format(epoch, ite, len_loader, self.counter, self.timer.avg, self.loss.avg,
                                           weighted_AUC * 100)
        )

    def write_dict(self, weighted_AUC):
        loss_dict = {
            'loss': self.loss.avg
        }

        if weighted_AUC > 0:
            loss_dict['weighted AUC'] = weighted_AUC

        return loss_dict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
