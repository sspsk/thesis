from data.eft_dataset import EFTDataset
import config

def test_eft_dataset():
    annot_files = [config.COCO14_ANNOT_FOLDER+"COCO2014-All-ver01.json"]
    img_folders = [config.COCO14_IMAGES_FOLDER]
    mydataset = EFTDataset(annot_filenames=annot_files,img_folders=img_folders)
    print("Dataset length:",len(mydataset))

    img,(cam,pose,shape) = mydataset[0]
    
    assert len(img.shape) == 3
    assert len(cam) == 3
    assert len(pose) == 24
    assert len(pose.shape) == 3
    assert len(shape) == 10

    print("EFT Dataset Completed Successfully.")