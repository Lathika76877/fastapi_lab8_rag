from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, EmailStr, Field


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Human-readable error explanation")

    model_config = ConfigDict(
        json_schema_extra={"example": {"detail": "User not found"}}
    )


class UserBase(BaseModel):
    email: EmailStr = Field(
        ...,
        description="Unique corporate or team email address used for authentication",
        examples=["ananya.sharma@agilisium.com"],
    )
    full_name: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Full display name of the team member",
        examples=["Ananya Sharma"],
    )
    role: str = Field(
        default="member",
        description="Assigned functional role (e.g., member, developer, ai_engineer, admin)",
        examples=["ai_engineer"],
    )
    is_active: bool = Field(
        default=True,
        description="Indicates whether the user account is active and permitted to authenticate",
    )


class UserCreate(UserBase):
    password: str = Field(
        ...,
        min_length=8,
        description="Plaintext password (minimum 8 characters); hashed via bcrypt before storage",
        examples=["SecureAI#2026"],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "email": "ananya.sharma@agilisium.com",
                "full_name": "Ananya Sharma",
                "role": "ai_engineer",
                "is_active": True,
                "password": "SecureAI#2026",
            }
        }
    )


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = Field(None, examples=["ananya.updated@agilisium.com"])
    full_name: Optional[str] = Field(
        None, min_length=2, max_length=100, examples=["Ananya S."]
    )
    role: Optional[str] = Field(None, examples=["lead_ai_engineer"])
    is_active: Optional[bool] = Field(None, examples=[True])

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "role": "lead_ai_engineer",
                "is_active": True,
            }
        }
    )


class UserResponse(UserBase):
    id: int = Field(..., description="Auto-incremented primary key identifier", examples=[1])
    created_at: Optional[datetime] = Field(
        default=None,
        description="UTC timestamp when the user record was created",
        examples=["2026-11-23T12:00:00Z"],
    )

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 1,
                "email": "ananya.sharma@agilisium.com",
                "full_name": "Ananya Sharma",
                "role": "ai_engineer",
                "is_active": True,
                "created_at": "2026-11-23T12:00:00Z",
            }
        },
    )


class Token(BaseModel):
    access_token: str = Field(..., description="Signed JWT Bearer access token")
    token_type: str = Field(default="bearer", description="OAuth2 token scheme type")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
            }
        }
    )


class TokenData(BaseModel):
    email: Optional[str] = None


class RequestLogResponse(BaseModel):
    id: int = Field(..., description="Unique audit log entry ID", examples=[1])
    user_id: int = Field(
        ..., description="ID of the authenticated user who made the request", examples=[1]
    )
    path: str = Field(..., description="Requested URL path", examples=["/users/me"])
    method: str = Field(
        ..., description="HTTP method (GET, POST, PATCH, DELETE)", examples=["GET"]
    )
    status_code: int = Field(
        ..., description="HTTP response status code returned", examples=[200]
    )
    timestamp: datetime = Field(
        ...,
        description="UTC timestamp when the request completed",
        examples=["2026-11-24T10:30:00Z"],
    )

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 1,
                "user_id": 1,
                "path": "/users/me",
                "method": "GET",
                "status_code": 200,
                "timestamp": "2026-11-24T10:30:00Z",
            }
        },
    )


class ChunkPreview(BaseModel):
    chunk_index: int = Field(..., examples=[0])
    content: str = Field(
        ...,
        examples=[
            "FastAPI is a modern, high-performance web framework for building APIs..."
        ],
    )
    metadata_json: Dict[str, Any] = Field(
        ..., examples=[{"source": "ai_handbook.md", "chunk_index": 0}]
    )


class IngestResponse(BaseModel):
    source_filename: str = Field(..., examples=["ai_handbook.md"])
    chunks_indexed: int = Field(
        ...,
        description="Total number of vector chunks persisted in pgvector",
        examples=[12],
    )
    embedding_dimension: int = Field(default=384, examples=[384])
    sample_chunks: List[ChunkPreview]


class VectorSearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        description="Natural language search query",
        examples=["How does OAuth2 token expiration work?"],
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of top matching chunks to retrieve",
        examples=[3],
    )
    source_filename: Optional[str] = Field(
        None,
        description="Optional metadata filter by exact document filename",
        examples=["ai_handbook.md"],
    )


class RetrievedChunk(BaseModel):
    id: int
    source_filename: str
    chunk_index: int
    content: str
    metadata_json: Dict[str, Any]
    similarity_score: float = Field(
        ...,
        description="Cosine similarity score (0.0 to 1.0, higher is more relevant)",
        examples=[0.8642],
    )


class VectorSearchResponse(BaseModel):
    query: str
    results: List[RetrievedChunk]