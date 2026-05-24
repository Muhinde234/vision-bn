"""
Patient endpoint tests.
"""
import pytest
from httpx import AsyncClient


async def _get_token(client: AsyncClient, role: str = "lab_technician") -> str:
    email = f"{role}@test.rw"
    await client.post("/api/v1/auth/register", json={
        "email": email, "full_name": "Test User",
        "password": "SecurePass1", "role": role,
    })
    resp = await client.post("/api/v1/auth/login", json={
        "email": email, "password": "SecurePass1"
    })
    return resp.json()["data"]["access_token"]


@pytest.mark.asyncio
async def test_create_patient(client: AsyncClient):
    token = await _get_token(client)
    response = await client.post(
        "/api/v1/patients",
        json={"full_name": "John Doe", "sex": "male"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 201
    data = response.json()["data"]
    assert data["full_name"] == "John Doe"
    assert data["patient_code"].startswith("VDX-")


@pytest.mark.asyncio
async def test_list_patients(client: AsyncClient):
    token = await _get_token(client, "lab_technician")
    response = await client.get(
        "/api/v1/patients",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert "items" in response.json()["data"]
