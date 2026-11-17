import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class AxonConfig:
    mode: str = os.getenv("AXONPG_MODE", "docker")
    db_host: str = os.getenv("AXONPG_DB_HOST", "127.0.0.1")
    db_port: int = int(os.getenv("AXONPG_DB_PORT", "5435"))
    db_name: str = os.getenv("AXONPG_DB_NAME", "axonpg")
    db_user: str = os.getenv("AXONPG_DB_USER", "axonpg")
    db_password: str = os.getenv("AXONPG_DB_PASSWORD", "axonpg_dev_pass")
    embed_provider: str = os.getenv("AXONPG_EMBED_PROVIDER", "fastembed")
    embed_model: str = os.getenv("AXONPG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    embed_dim: int = int(os.getenv("AXONPG_EMBED_DIM", "384"))
    embed_api_key: str = os.getenv("AXONPG_EMBED_API_KEY", "")
    embed_base_url: str = os.getenv("AXONPG_EMBED_BASE_URL", "")
