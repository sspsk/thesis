from data.eval_dataset import Dataset_3DPW
from torch.utils.data import DataLoader

def test_3dpw_dataset():
    dataset = Dataset_3DPW()
    dataloader = DataLoader(dataset,batch_size=32,shuffle=False)
    
    print("Dataset length:",len(dataset))

    for batch in dataloader:
        break

    assert 'img' in batch
    assert 'pose' in batch
    assert 'shape' in batch

    img = batch['img']
    pose = batch['pose']
    shape = batch['shape']

    assert img.shape[0] == 32
    assert len(img.shape) == 4
    assert len(pose.shape) == 2
    assert pose.shape[1] == 72
    assert len(shape.shape) == 2
    assert shape.shape[1] == 10

    print("3DPW Dataset tests completed successfully.")
