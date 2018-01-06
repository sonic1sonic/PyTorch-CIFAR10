from torch.utils.data import DataLoader

class CifarDataloader(DataLoader):

    def __init__(self, *args, **kwargs):
        super(CifarDataloader, self).__init__(*args, **kwargs)
