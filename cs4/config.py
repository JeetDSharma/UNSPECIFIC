"""
Central configuration management for CS4.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Central configuration manager."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    LOGS_DIR = PROJECT_ROOT / "logs"
    JOBS_DIR = PROJECT_ROOT / "jobs"
    CONFIGS_DIR = PROJECT_ROOT / "configs"
    OUTPUTS_DIR = DATA_DIR / "outputs"
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
    TOGETHERAI_API_KEY = os.getenv("TOGETHERAI_API_KEY")
    
    # Default models per task - single source of truth for each pipeline stage
    DEFAULT_MERGE_MODEL = "gpt-4.1-mini"           # Blog merging
    DEFAULT_CONSTRAINT_MODEL = "gpt-4.1-mini"      # Constraint generation
    DEFAULT_BASE_GEN_MODEL = "gpt-4.1-mini"        # Base content generation
    DEFAULT_FITTING_MODEL = "gpt-4.1-mini"         # Constraint fitting
    DEFAULT_EVALUATION_MODEL = "claude-sonnet-4-20250514"      # Evaluation
    
    # Fallback default model (if task-specific not specified)
    DEFAULT_MODEL = "gpt-4.1-mini"
    
    # Constraint parameters
    NUM_CONSTRAINTS = 39
    
    # Similarity thresholds
    SIMILAR_THRESHOLD = 0.75
    DISSIMILAR_THRESHOLD = 0.40
    
    # LLM API retry parameters (for API calls)
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0  # seconds
    
    @classmethod
    def load_domain_config(cls, domain: str) -> Dict[str, Any]:
        """Load domain-specific configuration."""
        config_path = cls.CONFIGS_DIR / "domains" / f"{domain}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Domain config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @classmethod
    def load_config(cls, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist."""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.LOGS_DIR.mkdir(exist_ok=True)
        cls.JOBS_DIR.mkdir(exist_ok=True)
        cls.OUTPUTS_DIR.mkdir(exist_ok=True)
        (cls.DATA_DIR / "raw").mkdir(exist_ok=True)
        (cls.DATA_DIR / "processed").mkdir(exist_ok=True)
        (cls.DATA_DIR / "embeddings").mkdir(exist_ok=True)
    
    @classmethod
    def get_api_key(cls, provider: str) -> Optional[str]:
        """Get API key for specified provider."""
        if provider.lower() == "openai":
            return cls.OPENAI_API_KEY
        elif provider.lower() in ("anthropic", "claude"):
            return cls.CLAUDE_API_KEY
        elif provider.lower() in ("together", "togetherai"):
            return cls.TOGETHERAI_API_KEY
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    @classmethod
    def validate_api_keys(cls):
        """Validate that required API keys are set."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in environment")
        if not cls.CLAUDE_API_KEY:
            raise ValueError("CLAUDE_API_KEY not set in environment")


# Initialize directories on import
Config.ensure_directories()
