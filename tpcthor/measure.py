""" Class to record & store training stats """
from uuid import uuid4


class Measure:
    def __init__(self, name=None):
        self._training_loss = []
        self._valid_stats = []
        self._lr = []
        self.name = name
        self.uuid = uuid4().hex

    def training_loss(self, batches, time, loss, lr):
        data = {'batches': batches, 'time': time, 'loss': loss, 'lr': lr}
        self._training_loss.append(data)
        print(f'{batches},{time:.1f},{lr:.5f},{loss:.5f}', flush=True)

    def valid_stats(self, batches, time, loss):
        data = {'batches': batches, 'time': time, 'loss': loss}
        self._valid_stats.append(data)
        fmt = f'VALID: {batches},{time:.1f},{loss:.5f}'
        print(fmt, flush=True)
