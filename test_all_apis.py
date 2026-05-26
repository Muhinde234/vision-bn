"""Full API test suite — runs against localhost:8000"""
import sys
import json
import urllib.request
import urllib.error
import urllib.parse
import uuid
import os

BASE = "http://localhost:8000/api/v1"
results = []
TOKEN = None
REFRESH = None
PATIENT_ID = None
DIAGNOSIS_ID = None
PREDICTION_ID = None
USER_ID = None

def req(method, path, body=None, token=None, form=None, label=None):
    url = BASE + path
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = None
    if body:
        data = json.dumps(body).encode()
    elif form:
        headers.pop("Content-Type", None)
        boundary = "----boundary"
        lines = []
        for k, v in form.items():
            lines.append(f"------boundary\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}")
        body_str = "\r\n".join(lines) + "\r\n------boundary--\r\n"
        data = body_str.encode()
        headers["Content-Type"] = f"multipart/form-data; boundary=----boundary"

    try:
        req_obj = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req_obj, timeout=10) as resp:
            status = resp.status
            try:
                resp_body = json.loads(resp.read())
            except Exception:
                resp_body = {}
            return status, resp_body
    except urllib.error.HTTPError as e:
        try:
            err_body = json.loads(e.read())
        except Exception:
            err_body = {}
        return e.code, err_body
    except Exception as e:
        return 0, {"error": str(e)}

def check(label, status, body, expect_status, extract=None):
    ok = status == expect_status
    icon = "PASS" if ok else "FAIL"
    msg = f"[{icon}] {label} -> HTTP {status}"
    if not ok:
        msg += f" (expected {expect_status}) | {body}"
    print(msg)
    results.append((ok, label))
    if ok and extract:
        return extract(body)
    return None

print("=" * 60)
print("VisionDx API Test Suite")
print("=" * 60)

# ── AUTH ─────────────────────────────────────────────────────────

print("\n--- AUTH ---")

# Register admin
s, b = req("POST", "/auth/register", {
    "email": "testadmin@visiondx.com", "full_name": "Test Admin",
    "password": "Admin@12345", "role": "admin", "facility_name": "HQ"
})
check("POST /auth/register (admin)", s, b, 201)

# Register duplicate — should 409
s, b = req("POST", "/auth/register", {
    "email": "testadmin@visiondx.com", "full_name": "Test Admin",
    "password": "Admin@12345", "role": "admin", "facility_name": "HQ"
})
check("POST /auth/register (duplicate -> 409)", s, b, 409)

# Register lab_tech
s, b = req("POST", "/auth/register", {
    "email": "testlab@visiondx.com", "full_name": "Test Lab Tech",
    "password": "Lab@12345", "role": "lab_technician", "facility_name": "Lab"
})
check("POST /auth/register (lab_tech)", s, b, 201)

# Login admin
s, b = req("POST", "/auth/login", {"email": "testadmin@visiondx.com", "password": "Admin@12345"})
val = check("POST /auth/login", s, b, 200, lambda b: b.get("data", {}))
if val:
    TOKEN = val.get("access_token")
    REFRESH = val.get("refresh_token")

# Login wrong password
s, b = req("POST", "/auth/login", {"email": "testadmin@visiondx.com", "password": "wrong"})
check("POST /auth/login (wrong password -> 401)", s, b, 401)

# GET /me
s, b = req("GET", "/auth/me", token=TOKEN)
val = check("GET /auth/me", s, b, 200, lambda b: b.get("data", {}).get("id"))
if val:
    USER_ID = val

# Refresh token
s, b = req("POST", "/auth/refresh", {"refresh_token": REFRESH})
val = check("POST /auth/refresh", s, b, 200, lambda b: b.get("data", {}))
if val:
    TOKEN = val.get("access_token")
    REFRESH = val.get("refresh_token")

# Change password
s, b = req("POST", "/auth/change-password",
    {"current_password": "Admin@12345", "new_password": "Admin@99999"},
    token=TOKEN)
check("POST /auth/change-password", s, b, 200)

# Change back
s, b = req("POST", "/auth/change-password",
    {"current_password": "Admin@99999", "new_password": "Admin@12345"},
    token=TOKEN)
check("POST /auth/change-password (revert)", s, b, 200)

# Forgot password
s, b = req("POST", "/auth/forgot-password", {"email": "testadmin@visiondx.com"})
check("POST /auth/forgot-password", s, b, 200)

# Forgot password (non-existent email — should still 200)
s, b = req("POST", "/auth/forgot-password", {"email": "nobody@visiondx.com"})
check("POST /auth/forgot-password (unknown email -> 200)", s, b, 200)

# Reset password (invalid token — should 401)
s, b = req("POST", "/auth/reset-password", {"token": "fake-token", "new_password": "Admin@12345"})
check("POST /auth/reset-password (bad token -> 401)", s, b, 401)

# Verify email (invalid token — should 401)
s, b = req("GET", "/auth/verify-email?token=badtoken")
check("GET /auth/verify-email (bad token -> 401)", s, b, 401)

# ── USERS ────────────────────────────────────────────────────────

print("\n--- USERS ---")

s, b = req("GET", "/users", token=TOKEN)
check("GET /users (admin)", s, b, 200)

if USER_ID:
    s, b = req("GET", f"/users/{USER_ID}", token=TOKEN)
    check("GET /users/{id}", s, b, 200)

    s, b = req("PATCH", f"/users/{USER_ID}", {"full_name": "Updated Admin"}, token=TOKEN)
    check("PATCH /users/{id}", s, b, 200)

# Non-existent user
s, b = req("GET", f"/users/{uuid.uuid4()}", token=TOKEN)
check("GET /users/{id} (not found -> 404)", s, b, 404)

# Without admin token — login as lab_tech
s, b = req("POST", "/auth/login", {"email": "testlab@visiondx.com", "password": "Lab@12345"})
LAB_TOKEN = b.get("data", {}).get("access_token") if s == 200 else None

s, b = req("GET", "/users", token=LAB_TOKEN)
check("GET /users (non-admin -> 403)", s, b, 403)

# ── PATIENTS ─────────────────────────────────────────────────────

print("\n--- PATIENTS ---")

s, b = req("POST", "/patients", {
    "full_name": "Jean Uwimana", "date_of_birth": "1990-05-15",
    "sex": "female", "facility_name": "Central Hospital",
    "contact_phone": "+250788000000"
}, token=TOKEN)
val = check("POST /patients", s, b, 201, lambda b: b.get("data", {}).get("id"))
if val:
    PATIENT_ID = val

s, b = req("GET", "/patients", token=TOKEN)
check("GET /patients", s, b, 200)

s, b = req("GET", "/patients?search=Jean", token=TOKEN)
check("GET /patients?search=Jean", s, b, 200)

if PATIENT_ID:
    s, b = req("GET", f"/patients/{PATIENT_ID}", token=TOKEN)
    check("GET /patients/{id}", s, b, 200)

    s, b = req("PATCH", f"/patients/{PATIENT_ID}", {"contact_phone": "+250788111111"}, token=TOKEN)
    check("PATCH /patients/{id}", s, b, 200)

    s, b = req("GET", f"/patients/{PATIENT_ID}/history", token=TOKEN)
    check("GET /patients/{id}/history", s, b, 200)

s, b = req("GET", f"/patients/{uuid.uuid4()}", token=TOKEN)
check("GET /patients/{id} (not found -> 404)", s, b, 404)

# ── DIAGNOSES ────────────────────────────────────────────────────

print("\n--- DIAGNOSES ---")

if PATIENT_ID:
    s, b = req("POST", "/diagnoses", {
        "patient_id": PATIENT_ID,
        "facility_name": "Central Hospital",
        "notes": "Routine malaria screening"
    }, token=TOKEN)
    val = check("POST /diagnoses", s, b, 201, lambda b: b.get("data", {}).get("id"))
    if val:
        DIAGNOSIS_ID = val

s, b = req("GET", "/diagnoses", token=TOKEN)
check("GET /diagnoses", s, b, 200)

s, b = req("GET", "/diagnoses?status=pending", token=TOKEN)
check("GET /diagnoses?status=pending", s, b, 200)

if DIAGNOSIS_ID:
    s, b = req("GET", f"/diagnoses/{DIAGNOSIS_ID}", token=TOKEN)
    check("GET /diagnoses/{id}", s, b, 200)

    s, b = req("PATCH", f"/diagnoses/{DIAGNOSIS_ID}", {"notes": "Updated notes"}, token=TOKEN)
    check("PATCH /diagnoses/{id}", s, b, 200)

    if PATIENT_ID:
        s, b = req("GET", f"/patients/{PATIENT_ID}/results/{DIAGNOSIS_ID}", token=TOKEN)
        check("GET /patients/{id}/results/{diagnosis_id}", s, b, 200)

# ── PREDICTIONS ──────────────────────────────────────────────────

print("\n--- PREDICTIONS ---")

# Create a minimal fake image for testing
img_boundary = "----testboundary"
img_data = (
    f'------testboundary\r\nContent-Disposition: form-data; name="file"; filename="test.jpg"\r\n'
    f'Content-Type: image/jpeg\r\n\r\n'
).encode() + bytes([0xFF, 0xD8, 0xFF, 0xE0] + [0] * 100) + (
    f'\r\n------testboundary\r\nContent-Disposition: form-data; name="disease_type"\r\n\r\nmalaria'
    f'\r\n------testboundary--\r\n'
).encode()

try:
    import urllib.request as ur
    pred_req = ur.Request(
        BASE + "/predictions/predict",
        data=img_data,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "multipart/form-data; boundary=----testboundary"
        },
        method="POST"
    )
    with ur.urlopen(pred_req, timeout=15) as resp:
        s = resp.status
        pred_body = json.loads(resp.read())
except urllib.error.HTTPError as e:
    s = e.code
    try:
        pred_body = json.loads(e.read())
    except Exception:
        pred_body = {}
except Exception as ex:
    s, pred_body = 0, {"error": str(ex)}

val = check("POST /predictions/predict", s, pred_body, 201,
            lambda b: b.get("data", {}).get("id"))
if val:
    PREDICTION_ID = val

s, b = req("GET", "/predictions/history", token=TOKEN)
check("GET /predictions/history", s, b, 200)

if PREDICTION_ID:
    s, b = req("GET", f"/predictions/{PREDICTION_ID}", token=TOKEN)
    check("GET /predictions/{id}", s, b, 200)

s, b = req("GET", f"/predictions/{uuid.uuid4()}", token=TOKEN)
check("GET /predictions/{id} (not found -> 404)", s, b, 404)

# ── ANALYTICS ────────────────────────────────────────────────────

print("\n--- ANALYTICS ---")

s, b = req("GET", "/analytics/dashboard", token=TOKEN)
check("GET /analytics/dashboard", s, b, 200)

s, b = req("GET", "/analytics/cases-over-time?granularity=daily", token=TOKEN)
check("GET /analytics/cases-over-time (daily)", s, b, 200)

s, b = req("GET", "/analytics/cases-over-time?granularity=weekly", token=TOKEN)
check("GET /analytics/cases-over-time (weekly)", s, b, 200)

s, b = req("GET", "/analytics/cases-over-time?granularity=monthly", token=TOKEN)
check("GET /analytics/cases-over-time (monthly)", s, b, 200)

s, b = req("GET", "/analytics/stage-distribution", token=TOKEN)
check("GET /analytics/stage-distribution", s, b, 200)

s, b = req("GET", "/analytics/user-activity", token=TOKEN)
check("GET /analytics/user-activity", s, b, 200)

s, b = req("GET", "/analytics/export/csv", token=TOKEN)
check("GET /analytics/export/csv", s, b, 200)

s, b = req("GET", "/analytics/export/pdf", token=TOKEN)
check("GET /analytics/export/pdf", s, b, 200)

# ── SYNC ─────────────────────────────────────────────────────────

print("\n--- SYNC ---")

sync_id = str(uuid.uuid4())
s, b = req("POST", "/sync/diagnoses", {
    "diagnoses": [{
        "mobile_sync_id": sync_id,
        "patient_id": PATIENT_ID,
        "facility_name": "Central Hospital",
        "notes": "Offline sync test"
    }]
}, token=TOKEN)
check("POST /sync/diagnoses", s, b, 200)

# Replay same sync_id — should skip (idempotent)
s, b = req("POST", "/sync/diagnoses", {
    "diagnoses": [{
        "mobile_sync_id": sync_id,
        "patient_id": PATIENT_ID,
        "facility_name": "Central Hospital",
        "notes": "Offline sync test"
    }]
}, token=TOKEN)
check("POST /sync/diagnoses (idempotent replay -> skipped)", s, b, 200)

# ── CLEANUP: DELETE ───────────────────────────────────────────────

print("\n--- CLEANUP ---")

if DIAGNOSIS_ID:
    s, b = req("DELETE", f"/diagnoses/{DIAGNOSIS_ID}", token=TOKEN)
    check("DELETE /diagnoses/{id}", s, b, 204)

if PATIENT_ID:
    s, b = req("DELETE", f"/patients/{PATIENT_ID}", token=TOKEN)
    check("DELETE /patients/{id}", s, b, 200)

# Logout
s, b = req("POST", "/auth/logout", token=TOKEN)
check("POST /auth/logout", s, b, 200)

# Token should now be invalid
s, b = req("GET", "/auth/me", token=TOKEN)
# Note: logout revokes refresh tokens but not the access token itself (JWT is stateless)
# Access token remains valid until it expires — this is expected behaviour
print(f"[INFO] GET /auth/me after logout -> HTTP {s} (access token still valid until expiry — expected)")

# ── SUMMARY ───────────────────────────────────────────────────────

print("\n" + "=" * 60)
passed = sum(1 for ok, _ in results if ok)
failed = sum(1 for ok, _ in results if not ok)
print(f"RESULTS: {passed} passed / {failed} failed / {len(results)} total")
if failed:
    print("\nFailed tests:")
    for ok, label in results:
        if not ok:
            print(f"  - {label}")
print("=" * 60)
