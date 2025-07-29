
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data, HeteroData
from typing import List, Optional, Tuple, NamedTuple, Union, Callable
import sys
import os






class ColumnNormalizeFeatures(BaseTransform):
    r"""column-normalizes the attributes given in :obj:`attrs` to sum-up to one.

    Args:
        attrs (List[str]): The names of attributes to normalize.
            (default: :obj:`["x"]`)
    """

    def __init__(self, attrs: List[str] = ["edge_attr"]):
        self.attrs = attrs

    def __call__(self, data: Union[Data, HeteroData]):
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                value.div_(value.sum(dim=0, keepdim=True).clamp_(min=1.))  # change dim
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_results = None

    def __call__(self, val_loss, results):
        if self.best_loss == None:
            self.best_loss = val_loss
            self.best_results = results
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
            # save best result
            self.best_results = results

        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"     INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('     INFO: Early stopping')
                self.early_stop = True

class Logger(object):
    def __init__(self, logdir: str):
        self.console = sys.stdout

        if logdir is not None:
            try:
                os.makedirs(logdir)
            except:
                pass
            self.log_file = open(logdir + '/log.txt', 'w')
        else:
            self.log_file = None
        sys.stdout = self
        sys.stderr = self

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.log_file is not None:
            self.log_file.write(msg)

    def flush(self):
        self.console.flush()
        if self.log_file is not None:
            self.log_file.flush()
            os.fsync(self.log_file.fileno())

    def close(self):
        self.console.close()
        if self.log_file is not None:
            self.log_file.close()