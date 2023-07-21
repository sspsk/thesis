#local imports
from data.eft_dataset import EFTDataset
import config

#imports
from torch.utils.data import DataLoader

def test_eft_dataset():
    annot_files = [config.COCO14_ANNOT_FOLDER+"COCO2014-All-ver01.json"]
    img_folders = [config.COCO14_IMAGES_FOLDER]
    mydataset = EFTDataset(annot_filenames=annot_files,img_folders=img_folders)
    print("Dataset length:",len(mydataset))

    sample = mydataset[0]
    
    assert len(sample['img']) == 3
    assert len(sample['cam']) == 3
    assert len(sample['pose']) == 24
    assert len(sample['pose'].shape) == 3
    assert len(sample['shape']) == 10

    print("EFT Dataset Completed Successfully.")

def test_eft_dataloader():
    annot_files = [config.COCO14_ANNOT_FOLDER+"COCO2014-All-ver01.json"]
    img_folders = [config.COCO14_IMAGES_FOLDER]
    mydataset = EFTDataset(annot_filenames=annot_files,img_folders=img_folders)
    loader = DataLoader(mydataset,batch_size=4,shuffle=False)
    for batch in loader:
        break

    assert 'img' in batch
    assert 'pose' in batch
    assert 'cam' in batch
    assert 'shape' in batch

    assert len(batch['img'].shape)==4
    assert len(batch['pose'].shape)==4
    assert len(batch['cam'].shape)==2
    assert len(batch['shape'].shape)==2
