import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.app.main import app
from backend.app.database.session import get_db
from backend.app.database.base import Base
from backend.app.core.config import settings

# Test database setup (using SQLite for simple testing without external deps in CI)
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="module")
def db():
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="module")
def client(db):
    def override_get_db():
        try:
            yield db
        finally:
            pass
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    del app.dependency_overrides[get_db]

def test_user_registration(client):
    user_data = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpassword"
    }
    response = client.post(f"{settings.API_V1_STR}/auth/register", json=user_data)
    assert response.status_code == 201
    assert response.json()["email"] == user_data["email"]

def test_user_login(client):
    # Registration first (or use a fixture)
    login_data = {
        "username": "testuser",
        "password": "testpassword"
    }
    response = client.post(
        f"{settings.API_V1_STR}/auth/login",
        data={"username": login_data["username"], "password": login_data["password"]}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert "refresh_token" in response.json()

def test_get_me(client):
    # Login to get token
    login_data = {
        "username": "testuser",
        "password": "testpassword"
    }
    login_res = client.post(
        f"{settings.API_V1_STR}/auth/login",
        data={"username": login_data["username"], "password": login_data["password"]}
    )
    token = login_res.json()["access_token"]
    
    response = client.get(
        f"{settings.API_V1_STR}/users/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json()["username"] == "testuser"
