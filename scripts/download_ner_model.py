import logging
import torch
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Model to Download ---
NER_MODEL_NAME = 'dslim/bert-base-NER'
# -------------------------

def download_model(model_name):
    logger.info(f"Attempting to download/cache NER model: '{model_name}'...")

    if torch.cuda.is_available():
        device_id = 0
        device_name = f"cuda:{device_id}"
    else:
        device_id = -1
        device_name = "cpu"
    logger.info(f"Will attempt initialization on device: {device_name}")

    try:
        _ = pipeline(
            "ner",
            model=model_name,
            grouped_entities=True, 
            device=device_id
        )
        logger.info(f"Model '{model_name}' is downloaded/cached and initialized successfully.")
    except Exception as e:
        logger.exception(f"Failed to download/initialize NER model '{model_name}'. Error: {e}")

if __name__ == "__main__":
    download_model(NER_MODEL_NAME)