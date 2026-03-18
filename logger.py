import csv


class CSVLogger:
    def __init__(self, path, n_groups):
        self.file = open(path, 'w')

        columns = ['epoch']
        for g in range(n_groups):
            columns += [
                f'loss_group_{g}',
                f'acc_group_{g}',
                f'adv_prob_group_{g}',
                f'n_group_{g}',
            ]
        columns += ['avg_loss', 'avg_acc', 'worst_group_acc']

        self.writer = csv.DictWriter(self.file, fieldnames=columns)
        self.writer.writeheader()

    def log(self, row):
        self.writer.writerow(row)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()