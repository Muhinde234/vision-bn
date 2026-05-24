"""
Auth endpoint integration tests.
"""
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_register_success(client: AsyncClient):
    response = await client.post("/api/v1/auth/register", json={
        "email": "tech@visiondx.rw",
        "full_name": "Alice Lab",
        "password": "SecurePass1",
        "role": "lab_technician",
    })
    assert response.status_code == 201
    data = response.json()
    assert data["success"] is True
    assert data["data"]["email"] == "tech@visiondx.rw"


@pytest.mark.asyncio
async def test_register_duplicate_email(client: AsyncClient):
    payload = {
        "email": "dup@visiondx.rw",
        "full_name": "Dup User",
        "password": "SecurePass1",
    }
    await client.post("/api/v1/auth/register", json=payload)
    response = await client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_login_success(client: AsyncClient):
    await client.post("/api/v1/auth/register", json={
        "email": "login@visiondx.rw",
        "full_name": "Login User",
        "password": "SecurePass1",
    })
    response = await client.post("/api/v1/auth/login", json={
        "email": "login@visiondx.rw",
        "password": "SecurePass1",
    })
    assert response.status_code == 200
    tokens = response.json()["data"]
    assert "access_token" in tokens
    assert "refresh_token" in tokens


@pytest.mark.asyncio
async def test_login_wrong_password(client: AsyncClient):
    await client.post("/api/v1/auth/register", json={
        "email": "wrongpw@visiondx.rw",
        "full_name": "Wrong PW",
        "password": "SecurePass1",
    })
    response = await client.post("/api/v1/auth/login", json={
        "email": "wrongpw@visiondx.rw",
        "password": "WrongPassword99",
    })
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_me_endpoint(client: AsyncClient):
    await client.post("/api/v1/auth/register", json={
        "email": "me@visiondx.rw",
        "full_name": "Me User",
        "password": "SecurePass1",
    })
    login_resp = await client.post("/api/v1/auth/login", json={
        "email": "me@visiondx.rw",
        "password": "SecurePass1",
    })
    token = login_resp.json()["data"]["access_token"]

    me_resp = await client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert me_resp.status_code == 200
    assert me_resp.json()["data"]["email"] == "me@visiondx.rw"
