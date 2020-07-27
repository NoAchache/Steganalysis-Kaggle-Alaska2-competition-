import os
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import csv

from utils import cfg, ModelUtils
from data_provider import TestDataLoader
from models import EndToEndNet


class Test:
    def __init__(self):
        self.model = EndToEndNet()

    def main(self):
        if cfg.device.type == 'cuda':
            torch.cuda.set_enabled_lms(True)
            cudnn.benchmark = True

        ModelUtils.load_model(self)

        self.model = self.model.to(cfg.device)

        test_data = TestDataLoader(img_root=os.path.join(cfg.data_path, 'test'))
        self.data_loader = data.DataLoader(test_data, batch_size=cfg.batch_size, shuffle=False,
                                       num_workers=cfg.num_workers)

        print(f'Testing {len(test_data) * cfg.batch_size} images')

        results = self._infere()
        self._write_results(results)


    def _infere(self):
        with torch.no_grad():
            self.model.eval()
            results = {}

            for i, (img, img_id) in enumerate(self.data_loader):
                img = img.to(cfg.device)
                classication_output = self.model(img)
                for pred_score, pred_id in zip(classication_output, img_id):
                    pred_score = 1 - torch.exp(pred_score[0]).item()
                    results[int(pred_id.numpy())] = pred_score

                if i % 10 == 0 and i > 0:
                    print('step {}'.format(i))
        return results

    def _write_results(self, results):
        ''' Write the results in the submission format of the competition '''

        columns = ['Id', 'Label']
        file_path = os.path.join(cfg.project_root, 'submission.csv')

        with open(file_path, 'w') as file:
            file_writer = csv.writer(file)
            file_writer.writerow(columns)

            for img_id, label in results.items():
                label = str(label)
                img_id = str(img_id)+'.jpg'
                missing_0s = 8 - len(img_id)
                img_id = missing_0s * '0' + img_id # 1.jpg --> 0001.jpg
                file_writer.writerow([img_id, label])

if __name__ == '__main__':
    test = Test()
    test.main()


