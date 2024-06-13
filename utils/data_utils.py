import os
from dotenv import load_dotenv
from datasets import load_dataset

from lib.logger import Logger

logger = Logger(__name__)

def format_example(example):
    return {"text": f"[INST] {example['instruction']} [/INST] {example['answer']}"}


class DatasetLoader:
    def __init__(self, dataset_path:str, use_hf:bool):
        load_dotenv()
        self.dataset_path = dataset_path
        self.use_hf = use_hf
        self.dataset_loaded = False
        self._load_dataset()

    def _load_dataset(self):
        # Load training split (you can process it here)
        self.train_dataset = load_dataset(
            self.dataset_path, use_auth_token=os.getenv("HF_AUTH_TOKEN"))
        self.train_dataset = self.train_dataset.map(format_example)
        self.train_dataset = self.train_dataset.remove_columns(
            ['instruction', 'answer'])
        self.data = self.train_dataset.train_test_split(test_size=0.2)
        # Set the dataset flag
        self.dataset_loaded = True

    def get_dataset(self):
        assert self.dataset_loaded, \
            "Dataset not loaded. Please run load_dataset() first."
        return self.data['train'], self.data['test']


