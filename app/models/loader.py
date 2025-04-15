import logging
import torch 
from sentence_transformers import SentenceTransformer
from transformers import pipeline 

logger = logging.getLogger(__name__)

# --- Model Configuration ---
SENTENCE_MODEL_NAME = 'all-MiniLM-L6-v2'
NER_MODEL_NAME = 'dslim/bert-base-NER'
# -------------------------

ner_model_instance = None
sentence_transformer_instance = None

def load_ner_model():
    """Loads the NER model using the transformers pipeline."""
    global ner_model_instance
    if ner_model_instance is None:
        # Determine device: Use GPU if available, otherwise CPU
        if torch.cuda.is_available():
            device_id = 0 
            device_name = f"cuda:{device_id}"
            logger.info(f"CUDA available. Loading NER model on GPU ({torch.cuda.get_device_name(device_id)})...")
        else:
            device_id = -1 
            device_name = "cpu"
            logger.info("CUDA not available. Loading NER model on CPU...")

        try:
            ner_model_instance = pipeline(
                "ner",
                model=NER_MODEL_NAME,
                grouped_entities=True,
                device=device_id 
            )
            logger.info(f"NER model ('{NER_MODEL_NAME}') loaded successfully on {device_name}.")
        except Exception as e:
            logger.exception(f"Failed to load NER model '{NER_MODEL_NAME}'. Error: {e}")
            ner_model_instance = None
    else:
        logger.debug("NER model already loaded.") 
    return ner_model_instance

def load_sentence_transformer():
    """Loads the Sentence Transformer model."""
    global sentence_transformer_instance
    if sentence_transformer_instance is None:
        logger.info(f"Loading Sentence Transformer model ('{SENTENCE_MODEL_NAME}')...")
        try:
            sentence_transformer_instance = SentenceTransformer(SENTENCE_MODEL_NAME)
            effective_device = sentence_transformer_instance.device
            logger.info(f"Sentence Transformer model ('{SENTENCE_MODEL_NAME}') loaded successfully on device: {effective_device}.")
        except Exception as e:
            logger.exception(f"Failed to load Sentence Transformer model '{SENTENCE_MODEL_NAME}'. Error: {e}")
            sentence_transformer_instance = None
    else:
        logger.debug("Sentence Transformer model already loaded.") 
    return sentence_transformer_instance

def startup_load_models():
    """Function to be called on application startup to preload models."""
    logger.info("Preloading ML models on application startup...")
    load_sentence_transformer()
    load_ner_model()
    logger.info("ML model preloading finished (check logs above for success/failure).")