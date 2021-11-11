from src.utility.config import Config
from src.loader import ILoader, DataLoader

# --- [Global Variable] ---
dataLoader: ILoader = None


def build_data_loader(config: Config) -> ILoader:
    global dataLoader
    if dataLoader is not None:
        return dataLoader

    params = {
        "target": config.target,
        "sample_size": config.sample_size,
        "sampling": config.sampling,
        "train_file_path": config.train_file_path,
        "test_file_path": config.test_file_path,
        "val_split": config.train_test_split[1],
    }

    loader = DataLoader(**params)
    dataLoader = loader

    return loader
