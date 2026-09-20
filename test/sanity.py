import os, torch, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from utils import get_val_pair_from_bin
from torchkit.backbone import get_model




print('1. Testing PyTorch CUDA availability...')
print('CUDA Available:', torch.cuda.is_available())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

print('2. Loading binary file...')
images, issame = get_val_pair_from_bin('/home/devashish_tripathi/projects/sanity_data/data_bdct', 'val_dummy.bin')
print(f'Loaded {len(images)} images successfully.')

print('3. Testing model instantiation...')
model = get_model('IR_50')([112, 112], input_channel=189).to(device)
print('Model on device successfully.')

print('4. Testing single forward pass with dummy tensor...')
from test_dctdp import images_to_batch
dummy = torch.randn(2, 3, 112, 112).to(device)
with torch.no_grad():
    x_dct = images_to_batch(dummy)
    out = model(x_dct)
print('Forward pass complete! Output shape:', out.shape)
