# utils/config.py
from pathlib import Path

class CFG:
    # ================== PATHS ==================
    # Update these paths when running locally
    DATA_DIR = Path('/kaggle/input/nih-chest-xrays')      # Change for local run
    IMAGE_DIR = DATA_DIR / 'images'
    LABELS_CSV = DATA_DIR / 'sample_labels.csv'           # or full Data_Entry_2017.csv

    OUTPUT_DIR = Path('/kaggle/working')                  # or './output'

    # ================== IMAGE ==================
    IMG_SIZE = 224                                        # 512 for better performance
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    # ================== TRAINING ==================
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    EPOCHS = 30
    LR = 1e-4
    WEIGHT_DECAY = 1e-5
    SEED = 42

    # ================== LABELS ==================
    DISEASE_LABELS = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Effusion', 'Emphysema', 'Fibrosis', 'Hernia',
        'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
        'Pneumonia', 'Pneumothorax'
    ]
    NUM_CLASSES = len(DISEASE_LABELS)

    # ================== FOCAL LOSS ==================
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0

    # ================== DEVICE ==================
    DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'


# Create output directory
CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
