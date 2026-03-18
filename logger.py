import csv
from dataset import labels


class CSVLogger:
    def __init__(self, path):
        self.file = open(path, 'w')

        columns = ['epoch']
        for label in labels:
            columns += [
                f'loss_group_{label}',
                f'accuracy_group_{label}',
                f'adv_prob_group_{label}',
                f'n_{label}',
            ]
        columns += ['avg_loss', 'avg_accuracy', 'worst_group_accuracy']

        self.writer = csv.DictWriter(self.file, fieldnames=columns)
        self.writer.writeheader()

    def log(self, row):
        self.writer.writerow(row)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()