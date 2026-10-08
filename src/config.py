IMG_SIZE    = 320
BATCH_SIZE  = 8

CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema',
    'Fibrosis', 'Pleural_Thickening', 'Hernia'
]
NUM_CLASSES = len(CLASSES)

THRESHOLDS = {
    'Atelectasis': 0.4729,
    'Cardiomegaly': 0.4002,
    'Effusion': 0.4951,
    'Infiltration': 0.5025,
    'Mass': 0.4228,
    'Nodule': 0.4243,
    'Pneumonia': 0.3641,
    'Pneumothorax': 0.4236,
    'Consolidation': 0.4092,
    'Edema': 0.3862,
    'Emphysema': 0.3938,
    'Fibrosis': 0.3751,
    'Pleural_Thickening': 0.4044,
    'Hernia': 0.3415
}

CHEXNET_AUCS = {
    'Atelectasis': 0.8094, 'Cardiomegaly': 0.9248, 'Effusion': 0.8638,
    'Infiltration': 0.7345, 'Mass': 0.8676, 'Nodule': 0.7802,
    'Pneumonia': 0.7680, 'Pneumothorax': 0.8887, 'Consolidation': 0.7901,
    'Edema': 0.8878, 'Emphysema': 0.9371, 'Fibrosis': 0.8047,
    'Pleural_Thickening': 0.8062, 'Hernia': 0.9164
}

# ============================================================
# RESEARCH / EXPLAINABILITY CONFIGURATION
# ============================================================
MC_DROPOUT_PASSES = 5
CONFIDENCE_THRESHOLD = 0.60
RELIABILITY_THRESHOLD = 0.50

# Weights for explanation reliability score formulation
# Score = W1*(1-Entropy) + W2*(1-Variance)
RELIABILITY_WEIGHT_ENTROPY = 0.5
RELIABILITY_WEIGHT_VARIANCE = 0.5