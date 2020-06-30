import csv
import os
from datetime import datetime

class CsvLogger:
    def __init__(self, output):
        self.fieldnames = ['epoch', 'time', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_iou']
        if output is None:
            raise ValueError
        if not os.path.exists(output):
            with open(output, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames = self.fieldnames)
                writer.writeheader()
        self.output = output


    def writeEpoch(self, epoch, train_loss, train_acc, val_loss, val_acc, val_iou):
        with open(self.output, 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch, datetime.now(), train_loss, train_acc, val_loss, val_acc, val_iou])