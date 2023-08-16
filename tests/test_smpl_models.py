from models.smpl import get_smpl_model

def test_basic_smpl():
    model = get_smpl_model(use_hands=True, use_feet_keypoints=True)
    output = model()
    print('Joints shape:',output.joints.shape)
    print("Test completed.")

def test_extra_smpl():
    model = get_smpl_model(use_hands=True, use_feet_keypoints=True,extra=True)
    output = model()
    print('Joints shape:',output.joints.shape)
    print("Test completed.")