from tensorboardX import SummaryWriter
import os


class TensorboardLogging(object):

    def __init__(self, log_path):
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.writer = SummaryWriter(log_path)

    def write_scalars(self, scalar_dict, n_iter, tag=None):
        for name, scalar in scalar_dict.items():
            if tag is not None:
                name = '/'.join([tag, name])
            self.writer.add_scalar(name, scalar, n_iter)

    def write_graph(self, model, image_tensor):
        self.writer.add_graph(model, [image_tensor], verbose=False)