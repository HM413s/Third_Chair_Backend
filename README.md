# Third Chair Backend

A FastAPI-based backend for the Third Chair application.

## Deployment on Railway

1. Create a new project on Railway
2. Connect your GitHub repository
3. Add the following environment variables in Railway:
   - `DATABASE_URL`: Your PostgreSQL database URL
   - `SECRET_KEY`: A secure secret key for JWT
   - `ALGORITHM`: JWT algorithm (default: HS256)
   - `ACCESS_TOKEN_EXPIRE_MINUTES`: Token expiration time
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `ALLOWED_ORIGINS`: Comma-separated list of allowed origins
   - `UPLOAD_DIR`: Directory for file uploads
   - `MILVUS_HOST`: Milvus host (if using)
   - `MILVUS_PORT`: Milvus port (if using)

4. Railway will automatically detect the Python application and deploy it using the Procfile

## Local Development

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and fill in your environment variables

4. Run the development server:
```bash
uvicorn main:app --reload
```

## API Documentation

Once the server is running, you can access:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`