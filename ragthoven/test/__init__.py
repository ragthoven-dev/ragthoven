import os
import pytest
from dotenv import load_dotenv

from ragthoven.utils import config_loader

@pytest.fixture(scope="session")
def setup_env():
    load_dotenv(override=True)

def load_config(cfg_path):
    load_dotenv(override=True)
    config = config_loader.load_config(cfg_path)
    
    if os.environ.get("OLLAMA_API_BASE", False):
        config.llm.model = "ollama/gemma3:1b"
        config.llm.base_url = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
    
    return config