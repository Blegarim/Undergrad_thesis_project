import torch
from torchvision import transforms
from models.Unified_Module import MultimodalModel
from scripts.PIE_sequence_Dataset import PIESequenceDataset
from utilities.visualize_utils import visualize_prediction

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'outputs/best_model_epoch.pth'
DATASET_PATH = 'data/PIE/set01'  # or your valid path

# Define your label names in correct order
LABEL_NAMES = ['gesture', 'look', 'action', 'cross']

# Load dataset (must match your training transform)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

dataset = PIESequenceDataset(
    data_dir=DATASET_PATH,
    label_key=None,
    label_mode='final',
    transform=transform,
    crop=True
)

# Load model
model = MultimodalModel(...)  # put your actual config here
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

# Visualize sample index (try 0, 1, 42, etc.)
visualize_prediction(dataset, model, idx=0, device=DEVICE, label_names=LABEL_NAMES)
