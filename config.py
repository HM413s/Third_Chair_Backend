import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Zilliz Cloud Configuration
ZILLIZ_CLOUD_API_KEY = "7fe4e8d09456ec9b1b78cdb8b86138a90fc0b36e2d1b76270d25370516ab55183bb3fa216667e19431dc4e05498d24f035bb54db"
ZILLIZ_CLUSTER_NAME = "TRADE_IDE"
ZILLIZ_CLOUD_ENDPOINT = "https://in03-409cc02a9c122d4.serverless.gcp-us-west1.cloud.zilliz.com"

# Zilliz/Milvus Configuration (fallback for local development)
ZILLIZ_HOST = os.getenv("ZILLIZ_HOST", "localhost")
ZILLIZ_PORT = os.getenv("ZILLIZ_PORT", "19530")
ZILLIZ_USER = os.getenv("ZILLIZ_USER", "")
ZILLIZ_PASSWORD = os.getenv("ZILLIZ_PASSWORD", "")
ZILLIZ_URI = os.getenv("ZILLIZ_URI", ZILLIZ_CLOUD_ENDPOINT)  # Use cloud endpoint by default

# Storage Configuration
STORAGE_DIR = None  # Disabled - storing only in Zilliz Cloud
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
ENABLE_LOCAL_BACKUP = False  # Disable local metadata backup


# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Document Processing Configuration
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))
print("MAX_FILE_SIZE",MAX_FILE_SIZE)

# Search Configuration
SEARCH_LIMIT = int(os.getenv("SEARCH_LIMIT", "5"))
SEARCH_EF = int(os.getenv("SEARCH_EF", "64"))  # HNSW search parameter 


