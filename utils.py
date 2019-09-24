import os
from torch.nn.modules.module import _addindent
import torch
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import Dataset
from models import INPUT_LENGTH
import numpy as np


class FsStorage:
    def get_bucket(self):
        return self

    def put(self, key, value):
        if len(key.split('/')) > 1:
            path = "/".join(key.split("/")[:-1])
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        with open(key, 'wb') as f:
            f.write(value)
    
    def get(self, key):
        with open(key, 'rb') as f:
            return f.read()

    def delete(self, key):
        return os.remove(key)

    # def ls(self):
    #     return os.listdir(self._prefix)


def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr 
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'   

    tmpstr = tmpstr + ')'
    return tmpstr


class PDFDataSet(Dataset):
    def __init__(self, df, data_path, first_n_byte=INPUT_LENGTH):
        self.df = df
        self.storage = FsStorage()
        self.data_path = data_path if data_path.endswith('/') or len(data_path) == 0 else data_path + '/'
        self.first_n_byte = first_n_byte

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cnt = self.storage.get((self.data_path + row['hash']))
        tmp_ = [i+1 for i in cnt[:self.first_n_byte]]
        tmp_ = tmp_+[0]*(self.first_n_byte-len(tmp_))
        return np.array(tmp_), np.array([row['verdict']])    


def regular_iterator(a, total=None):
    for elem in a:
        yield elem


def predict(model, dataloader, device, verbose=True):
    val_pred = []
    val_label = []
    iterarator = tqdm if verbose else regular_iterator
    for _,val_batch_data in iterarator(enumerate(dataloader), total=len(dataloader)):
        cur_batch_size = val_batch_data[0].size(0)

        exe_input = val_batch_data[0].to(device)
        exe_input = Variable(exe_input.long(), requires_grad=False)

        label = val_batch_data[1].to(device)
        label = Variable(label.float(), requires_grad=False)

        pred = model(exe_input)
        val_pred.extend(pred.cpu().data)
        val_label.extend(label)
    return np.array(val_pred), np.array(val_label)