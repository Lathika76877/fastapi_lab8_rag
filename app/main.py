from datetime import timedelta
from typing import List, Optional
from fastapi import Depends, FastAPI, File, HTTPException, Path, Query, Response, UploadFile, status
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from sqlalchemy.orm import Session

from . import cache, crud, models, rag_service, schemas, security
from .database import engine, get_db
from .middleware import AuthRequestLoggingMiddleware

with engine.begin() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
models.Base.metadata.create_all(bind=engine)
with engine.begin() as conn:
    conn.execute(
        text(
            "CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw "
            "ON document_chunks USING hnsw (embedding vector_cosine_ops)"
        )
    )

TAGS_METADATA = [
    {
        "name": "Authentication",
        "description": "OAuth2 Password Flow endpoints for exchanging user credentials for signed JWT Bearer tokens.",
    },
    {
        "name": "Users",
        "description": "User lifecycle management: self-registration, authenticated profile inspection, and CRUD operations.",
    },
    {
        "name": "Monitoring",
        "description": "Auth-aware request audit logs captured automatically by HTTP middleware.",
    },
    {
        "name": "RAG Knowledge Base",
        "description": "Document ingestion, recursive text chunking, pgvector indexing, and semantic retrieval.",
    },
]

API_DESCRIPTION = """
### AI Engineering Track — Core Backend Service

This service provides **OAuth2-secured User Management & RAG Knowledge Base Indexing** built with FastAPI, PostgreSQL (`pgvector`), SQLAlchemy, Redis, and LangChain.

#### Authentication Workflow
1. Register an account via **`POST /users/`**.
2. Click the **Authorize** button (top right) or call **`POST /login`** with your email as `username` and password.
3. All padlock-protected endpoints automatically receive `Authorization: Bearer <token>`.
"""

app = FastAPI(
    title="AI Engineering Track — User & RAG API",
    description=API_DESCRIPTION,
    version="3.0.0",
    contact={
        "name": "AI Engineering & KMD Team",
        "email": "ai-track-support@example.com",
    },
    license_info={
        "name": "Internal Training License",
    },
    openapi_tags=TAGS_METADATA,
    docs_url=None,
    redoc_url="/redoc",
)

app.add_middleware(AuthRequestLoggingMiddleware)
app.mount("/static", StaticFiles(directory="static", check_dir=False), name="static")


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html() -> HTMLResponse:
    html = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="AI Track API Portal — Interactive Swagger Docs",
        swagger_favicon_url="/static/team-logo.svg",
        swagger_ui_parameters={
            "persistAuthorization": True,
            "displayRequestDuration": True,
            "filter": True,
            "docExpansion": "list",
        },
    )
    custom_branding = """
    <style>
      .swagger-ui .topbar { background-color: #0F172A; padding: 10px 0; }
      .swagger-ui .topbar .link img { display: none; }
      .swagger-ui .topbar .link::before {
        content: "";
        display: inline-block;
        width: 220px;
        height: 44px;
        background: url('/static/team-logo.svg') no-repeat center center;
        background-size: contain;
      }
    </style>
    """
    body = html.body.decode("utf-8").replace("</head>", f"{custom_branding}</head>")
    return HTMLResponse(content=body, status_code=200)


# 0. AUTHENTICATION: LOGIN & TOKEN ISSUANCE
@app.post(
    "/login",
    response_model=schemas.Token,
    tags=["Authentication"],
    summary="Authenticate user & issue JWT access token",
    description=(
        "Validates form-encoded email (`username`) and `password` credentials against PostgreSQL "
        "and returns a time-bound **JWT Bearer access token** valid for 30 minutes."
    ),
    responses={
        401: {"model": schemas.ErrorResponse, "description": "Invalid email or password"},
    },
)
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    user = crud.authenticate_user(
        db, email=form_data.username, password=form_data.password
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=security.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = security.create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


# 1. PUBLIC REGISTRATION
@app.post(
    "/users/",
    response_model=schemas.UserResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Users"],
    summary="Register a new user account",
    description=(
        "Creates a new user record in PostgreSQL. Validates email uniqueness, hashes the plaintext "
        "password with **bcrypt**, and returns the sanitized user profile."
    ),
    responses={
        400: {"model": schemas.ErrorResponse, "description": "Email address already registered"},
    },
)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    return crud.create_user(db=db, user=user)


# 2. PROTECTED: CURRENT AUTHENTICATED USER
@app.get(
    "/users/me",
    response_model=schemas.UserResponse,
    tags=["Users"],
    summary="Get current authenticated user's profile",
    description="Decodes the active Bearer JWT token and returns the profile of the logged-in user.",
    responses={
        401: {"model": schemas.ErrorResponse, "description": "Missing or invalid Bearer token"},
    },
)
def read_users_me(
    current_user: models.User = Depends(security.get_current_active_user),
):
    return current_user


# 3. PROTECTED: READ ALL USERS
@app.get(
    "/users/",
    response_model=List[schemas.UserResponse],
    tags=["Users"],
    summary="List registered users (paginated)",
    description="Retrieves a paginated list of users from PostgreSQL. Requires a valid Bearer token.",
    responses={
        401: {"model": schemas.ErrorResponse, "description": "Unauthorized access"},
    },
)
def read_users(
    skip: int = Query(0, ge=0, description="Number of records to skip for pagination", examples=[0]),
    limit: int = Query(100, ge=1, le=500, description="Maximum number of user records to return", examples=[25]),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_active_user),
):
    return crud.get_users(db, skip=skip, limit=limit)


# 4. PROTECTED: READ SINGLE USER (REDIS CACHE-ASIDE)
@app.get(
    "/users/{user_id}",
    response_model=schemas.UserResponse,
    tags=["Users"],
    summary="Retrieve a specific user by ID (Redis Cached)",
    description=(
        "Implements the **Cache-Aside** pattern using Redis (`user:profile:{user_id}`, TTL 300s). "
        "Inspect the `X-Cache` response header (`HIT` vs `MISS`) to verify cache behavior."
    ),
    responses={
        401: {"model": schemas.ErrorResponse, "description": "Unauthorized access"},
        404: {"model": schemas.ErrorResponse, "description": "User ID not found"},
    },
)
def read_user(
    response: Response,
    user_id: int = Path(..., gt=0, description="Unique integer ID of the user", examples=[1]),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_active_user),
):
    cached_user = cache.get_cached_user(user_id)
    if cached_user is not None:
        response.headers["X-Cache"] = "HIT"
        return cached_user

    db_user = crud.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")

    user_schema = schemas.UserResponse.model_validate(db_user)
    cache.set_cached_user(user_id, user_schema.model_dump(mode="json"))
    response.headers["X-Cache"] = "MISS"
    return user_schema


# 5. PROTECTED: UPDATE USER (INVALIDATES CACHE)
@app.patch(
    "/users/{user_id}",
    response_model=schemas.UserResponse,
    tags=["Users"],
    summary="Partially update an existing user (Invalidates Cache)",
    description="Updates user fields in PostgreSQL and invalidates `user:profile:{user_id}` in Redis.",
    responses={
        400: {"model": schemas.ErrorResponse, "description": "Updated email already in use"},
        401: {"model": schemas.ErrorResponse, "description": "Unauthorized access"},
        404: {"model": schemas.ErrorResponse, "description": "User ID not found"},
    },
)
def update_user(
    user_update: schemas.UserUpdate,
    user_id: int = Path(..., gt=0, description="Unique integer ID of the user to update", examples=[1]),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_active_user),
):
    db_user = crud.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    if user_update.email and user_update.email != db_user.email:
        if crud.get_user_by_email(db, email=user_update.email):
            raise HTTPException(status_code=400, detail="Email already in use")

    updated_user = crud.update_user(db=db, db_user=db_user, user_update=user_update)
    cache.invalidate_user_cache(user_id)
    return updated_user


# 6. PROTECTED: DELETE USER (INVALIDATES CACHE)
@app.delete(
    "/users/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Users"],
    summary="Delete a user account (Invalidates Cache)",
    description="Deletes the user from PostgreSQL and evicts `user:profile:{user_id}` from Redis.",
    responses={
        401: {"model": schemas.ErrorResponse, "description": "Unauthorized access"},
        404: {"model": schemas.ErrorResponse, "description": "User ID not found"},
    },
)
def delete_user(
    user_id: int = Path(..., gt=0, description="Unique integer ID of the user to delete", examples=[1]),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_active_user),
):
    db_user = crud.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    crud.delete_user(db=db, db_user=db_user)
    cache.invalidate_user_cache(user_id)
    return None


# 7. PROTECTED: INSPECT AUTHENTICATED REQUEST AUDIT LOGS
@app.get(
    "/logs/",
    response_model=List[schemas.RequestLogResponse],
    tags=["Monitoring"],
    summary="Retrieve authenticated request audit logs",
    description=(
        "Returns recent HTTP requests made by authenticated users, recorded automatically by "
        "`AuthRequestLoggingMiddleware` into PostgreSQL."
    ),
    responses={
        401: {"model": schemas.ErrorResponse, "description": "Unauthorized access"},
    },
)
def read_request_logs(
    user_id: Optional[int] = Query(
        None, gt=0, description="Optional filter to view logs for a specific user ID", examples=[1]
    ),
    skip: int = Query(0, ge=0, description="Number of log records to skip", examples=[0]),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of log records to return", examples=[20]),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_active_user),
):
    return crud.get_request_logs(db, skip=skip, limit=limit, user_id=user_id)


# 8. PROTECTED: INGEST DOCUMENT INTO PGVECTOR
@app.post(
    "/rag/ingest",
    response_model=schemas.IngestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["RAG Knowledge Base"],
    summary="Upload, chunk & embed a document into pgvector",
    description=(
        "Accepts a `.txt`, `.md`, or `.pdf` file, splits text into overlapping 500-character chunks via "
        "LangChain `RecursiveCharacterTextSplitter`, generates 384-dim embeddings, and stores vectors in PostgreSQL."
    ),
    responses={
        400: {"model": schemas.ErrorResponse, "description": "Unsupported or empty file"},
        401: {"model": schemas.ErrorResponse, "description": "Unauthorized access"},
    },
)
async def ingest_document_endpoint(
    file: UploadFile = File(
        ..., description="Knowledge base document (.txt, .md, or .pdf)"
    ),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_active_user),
):
    chunks = await rag_service.ingest_document(
        db=db, file=file, user_id=current_user.id
    )
    sample_previews = [
        schemas.ChunkPreview(
            chunk_index=c.chunk_index,
            content=c.content,
            metadata_json=c.metadata_json,
        )
        for c in chunks[:3]
    ]
    return schemas.IngestResponse(
        source_filename=file.filename or "untitled.txt",
        chunks_indexed=len(chunks),
        embedding_dimension=rag_service.EMBEDDING_DIM,
        sample_chunks=sample_previews,
    )


# 9. PROTECTED: SEMANTIC VECTOR SEARCH PREVIEW
@app.post(
    "/rag/search",
    response_model=schemas.VectorSearchResponse,
    tags=["RAG Knowledge Base"],
    summary="Retrieve top-k semantically similar chunks from pgvector",
    description=(
        "Embeds the user's natural language query and executes cosine similarity search (`<=>`) "
        "against `document_chunks` with optional `source_filename` metadata filtering."
    ),
    responses={
        401: {"model": schemas.ErrorResponse, "description": "Unauthorized access"},
    },
)
def search_knowledge_base(
    payload: schemas.VectorSearchRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_active_user),
):
    matches = rag_service.search_similar_chunks(
        db=db,
        query=payload.query,
        top_k=payload.top_k,
        source_filename=payload.source_filename,
    )
    return schemas.VectorSearchResponse(query=payload.query, results=matches)