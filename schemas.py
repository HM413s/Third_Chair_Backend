from pydantic import BaseModel,EmailStr
from typing import Optional,List,Dict,Any






# User schema
class UserCreate(BaseModel):
    username: str
    password: str
    email: EmailStr


# User response schema
class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    is_active: bool

    class Config:
        from_attributes = True  # This allows the model to work with SQLAlchemy models


# Login schema
class Login(BaseModel):
    email: EmailStr
    password: str


# Token schema for authentication
class Token(BaseModel):
    access_token: str
    token_type: str


# TokenData schema to hold data related to the user (usually after token verification)
class TokenData(BaseModel):
    id: Optional[int] = None



#############################
# Chat Schemas
#############################



class ChatResponse(BaseModel):
    response: str
    images: List[Dict[str, str]]
    metadata: Dict[str, int]

class StatsResponse(BaseModel):
    total_documents: int
    document_types: Dict[str, int]
    sources: Dict[str, int]
    creation_dates: Dict[str, int]

class UploadResponse(BaseModel):
    message: str
    documents_processed: int

class DeleteResponse(BaseModel):
    message: str

# Multi-document schemas
class MultiUploadResponse(BaseModel):
    message: str
    session_id: str
    documents_processed: List[Dict[str, Any]]
    total_documents: int

class ComparisonRequest(BaseModel):
    query: str
    session_id: str = None
    document_sources: List[str] = None

class ComparisonResponse(BaseModel):
    comparison_analysis: Dict[str, Any]
    documents_analyzed: List[str]
    session_id: str = None

class DocumentSession(BaseModel):
    session_id: str
    created_at: str
    documents: List[str]
    total_chunks: int
    session_name: str = None

class SessionsResponse(BaseModel):
    sessions: Dict[str, DocumentSession]
    total_sessions: int