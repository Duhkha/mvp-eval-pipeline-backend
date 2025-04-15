import spacy
import logging
import platform

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Model to Download/Verify ---
SPACY_MODEL_NAME = "en_core_web_trf"
# --------------------------------

def download_or_verify_model(model_name):
    logger.info(f"Attempting to load/verify spaCy model: '{model_name}'...")
    try:
        if not spacy.util.is_package(model_name):
             logger.error(f"Model '{model_name}' package not found by spaCy.")
             logger.error("Please ensure you installed it correctly, e.g., via:")
             logger.error("pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.7.3/en_core_web_trf-3.7.3-py3-none-any.whl")
             return False

        nlp = spacy.load(model_name)
        logger.info(f"Successfully loaded spaCy model '{model_name}'.")
        logger.info(f"Pipeline components: {nlp.pipe_names}")
        try:
             from thinc.api import prefer_gpu, require_gpu
             if prefer_gpu():
                  logger.info("spaCy is configured to prefer GPU.")
             if require_gpu(): 
                  logger.info("spaCy successfully required GPU.")
        except ImportError:
             logger.info("Could not import thinc.api extras for GPU check (might need spacy[cuda] variant if installed differently)")
        except Exception as gpu_err:
              logger.warning(f"GPU check/requirement failed: {gpu_err}")

        return True
    except OSError as e:
         logger.error(f"OSError loading model '{model_name}'. Have the underlying transformer models been downloaded by huggingface cache? Error: {e}")
         logger.error("You might need to ensure internet connectivity or check huggingface cache.")
         return False
    except Exception as e:
        logger.exception(f"Failed to load/verify spaCy model '{model_name}'. Error: {e}")
        return False

if __name__ == "__main__":
    if platform.system() == 'Windows':
         os.environ["TOKENIZERS_PARALLELISM"] = "false"

    download_or_verify_model(SPACY_MODEL_NAME)