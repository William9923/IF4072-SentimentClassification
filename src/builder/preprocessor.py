from src.utility.config import Config
from src.preprocessor import IPreprocessor, TextPreprocessor

# --- [Global Variable] ---
textPrep: IPreprocessor = None


def build_text_prep(config: Config) -> IPreprocessor:
    global textPrep
    if textPrep is not None:
        return textPrep

    params = {
        "component": config.preprocessor_component,
    }

    prep = TextPreprocessor(**params)
    textPrep = prep

    return prep
