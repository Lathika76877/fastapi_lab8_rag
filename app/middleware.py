import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from .database import SessionLocal
from . import crud

logger = logging.getLogger("uvicorn.error")


class AuthRequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request.state.user = None

        response = await call_next(request)

        authenticated_user = getattr(request.state, "user", None)
        if authenticated_user is not None:
            db = SessionLocal()
            try:
                crud.create_request_log(
                    db=db,
                    user_id=authenticated_user.id,
                    path=request.url.path,
                    method=request.method,
                    status_code=response.status_code,
                )
            except Exception as exc:
                logger.error(f"Failed to persist RequestLog: {exc}")
            finally:
                db.close()

        return response