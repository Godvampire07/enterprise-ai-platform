from typing import Optional
from pydantic import BaseModel

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str

class TokenPayload(BaseModel):
    sub: Optional[str] = None
    type: Optional[str] = None

class Login(BaseModel):
    username: str
    password: str

class RefreshTokenRequest(BaseModel):
    refresh_token: str
