""" Class to record & store training stats """
from uuid import uuid4


class Measure:
    def __init__(self, name=None):
        self._training_loss = []
        self._valid_stats = []
        self._lr = []
        self.name = name
        self.uuid = uuid4().hex

    def training_loss(self, epoch, time, loss, lr):
        data = (epoch, time, loss, lr)
        data = {'epoch': epoch, 'time': time, 'loss': loss, 'lr': lr}
        self._training_loss.append(data)
        print(f'{epoch:.2f},{time:.1f},{loss:.5f},{lr:.5f}', flush=True)

    def valid_stats(self, epoch, time, loss):
        data = (epoch, time, loss)
        data = {'epoch': epoch, 'time': time, 'loss': loss}
        self._valid_stats.append(data)
        fmt = f'VALID: {epoch:.2f},{time:.1f},{loss:.5f}'
        print(fmt, flush=True)
