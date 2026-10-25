import logging
import secrets
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field
from sqlmodel import Session, select
from src.database.models import User, UserSession, get_session, hash_password, verify_password

logger = logging.getLogger("vibesync.api.auth")
router = APIRouter(prefix="/api/v1/auth", tags=["User Authentication & Accounts"])


# Request & Response Schemas
class SignupRequest(BaseModel):
    username: str = Field(min_length=3, max_length=30)
    email: str = Field(min_length=5, max_length=100)
    password: str = Field(min_length=6, max_length=100)


class LoginRequest(BaseModel):
    username_or_email: str
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: str


class AuthResponse(BaseModel):
    status: str
    message: str
    user: UserResponse
    token: str


# Authentication Dependency Helper
def get_current_user_optional(
    authorization: Optional[str] = Header(default=None),
    x_user_token: Optional[str] = Header(default=None),
    session: Session = Depends(get_session),
) -> Optional[User]:
    """
    Extract authenticated User if valid Bearer or X-User-Token header is provided.
    Returns None for guest (unauthenticated) requests.
    """
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:].strip()
    elif x_user_token:
        token = x_user_token.strip()

    if not token:
        return None

    user_sess = session.exec(select(UserSession).where(UserSession.token == token)).first()
    if not user_sess:
        return None

    return session.get(User, user_sess.user_id)


def get_current_user_required(
    current_user: Optional[User] = Depends(get_current_user_optional),
) -> User:
    """Dependency that raises 401 if user is not authenticated."""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please log in to access this feature.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user


@router.post("/signup", response_model=AuthResponse)
def signup(payload: SignupRequest, session: Session = Depends(get_session)):
    """Register a new user account."""
    clean_username = payload.username.strip().lower()
    clean_email = payload.email.strip().lower()

    # Check for existing username
    existing_user = session.exec(select(User).where(User.username == clean_username)).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username is already taken.")

    # Check for existing email
    existing_email = session.exec(select(User).where(User.email == clean_email)).first()
    if existing_email:
        raise HTTPException(status_code=400, detail="An account with this email already exists.")

    new_user = User(
        username=clean_username,
        email=clean_email,
        password_hash=hash_password(payload.password),
    )
    session.add(new_user)
    session.commit()
    session.refresh(new_user)

    if new_user.id is None:
        raise HTTPException(status_code=500, detail="Failed to initialize user record.")
    user_id: int = new_user.id

    # Issue session token
    token = secrets.token_hex(32)
    user_sess = UserSession(token=token, user_id=user_id)
    session.add(user_sess)
    session.commit()

    return AuthResponse(
        status="success",
        message="Account created successfully.",
        user=UserResponse(id=user_id, username=new_user.username, email=new_user.email),
        token=token,
    )


@router.post("/login", response_model=AuthResponse)
def login(payload: LoginRequest, session: Session = Depends(get_session)):
    """Authenticate existing user by username or email."""
    login_id = payload.username_or_email.strip().lower()

    user = session.exec(select(User).where((User.username == login_id) | (User.email == login_id))).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    if user.id is None:
        raise HTTPException(status_code=500, detail="User record is missing primary key.")
    user_id: int = user.id

    token = secrets.token_hex(32)
    user_sess = UserSession(token=token, user_id=user_id)
    session.add(user_sess)
    session.commit()

    return AuthResponse(
        status="success",
        message="Logged in successfully.",
        user=UserResponse(id=user_id, username=user.username, email=user.email),
        token=token,
    )


@router.get("/me", response_model=UserResponse)
def get_me(user: User = Depends(get_current_user_required)):
    """Retrieve currently authenticated user profile."""
    if user.id is None:
        raise HTTPException(status_code=500, detail="Invalid user profile.")
    return UserResponse(id=user.id, username=user.username, email=user.email)


@router.post("/logout")
def logout(
    authorization: Optional[str] = Header(default=None),
    x_user_token: Optional[str] = Header(default=None),
    session: Session = Depends(get_session),
):
    """Log out and revoke current session token."""
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:].strip()
    elif x_user_token:
        token = x_user_token.strip()

    if token:
        sess = session.exec(select(UserSession).where(UserSession.token == token)).first()
        if sess:
            session.delete(sess)
            session.commit()

    return {"status": "success", "message": "Logged out successfully."}
