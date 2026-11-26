from typing import Optional
from sqlalchemy.orm import Session
from . import models, schemas, security


def get_user(db: Session, user_id: int) -> Optional[models.User]:
    return db.query(models.User).filter(models.User.id == user_id).first()


def get_user_by_email(db: Session, email: str) -> Optional[models.User]:
    return db.query(models.User).filter(models.User.email == email).first()


def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()


def create_user(db: Session, user: schemas.UserCreate) -> models.User:
    hashed_password = security.get_password_hash(user.password)
    db_user = models.User(
        email=user.email,
        hashed_password=hashed_password,
        full_name=user.full_name,
        role=user.role,
        is_active=user.is_active,
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def authenticate_user(
    db: Session, email: str, password: str
) -> Optional[models.User]:
    user = get_user_by_email(db, email=email)
    if not user:
        return None
    if not security.verify_password(password, user.hashed_password):
        return None
    return user


def update_user(
    db: Session, db_user: models.User, user_update: schemas.UserUpdate
) -> models.User:
    update_data = user_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_user, field, value)
    db.commit()
    db.refresh(db_user)
    return db_user


def delete_user(db: Session, db_user: models.User) -> None:
    db.delete(db_user)
    db.commit()


def create_request_log(
    db: Session, user_id: int, path: str, method: str, status_code: int
) -> models.RequestLog:
    db_log = models.RequestLog(
        user_id=user_id,
        path=path,
        method=method,
        status_code=status_code,
    )
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    return db_log


def get_request_logs(
    db: Session, skip: int = 0, limit: int = 100, user_id: Optional[int] = None
):
    query = db.query(models.RequestLog)
    if user_id is not None:
        query = query.filter(models.RequestLog.user_id == user_id)
    return (
        query.order_by(models.RequestLog.timestamp.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )