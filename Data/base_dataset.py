from Logging.logging import Logger
from Data.utils import some_utils
class BaseDataset:
    def __init__(self,**kwargs):
        some_utils()

    def __len__(self):
        pass

    def __getitem__(self,idx):
        pass

