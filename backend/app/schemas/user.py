from typing import Optional
from pydantic import BaseModel, EmailStr, ConfigDict
from datetime import datetime

class UserBase(BaseModel):
    username: str
    email: EmailStr
    role: Optional[str] = "user"
    is_active: Optional[bool] = True

class UserCreate(UserBase):
    password: str

class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None

class UserInDBBase(UserBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

class User(UserInDBBase):
    pass

class UserInDB(UserInDBBase):
    hashed_password: str
