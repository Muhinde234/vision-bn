# VisionDx – Next.js ↔ FastAPI Integration Guide

## Base URL

| Environment | URL |
|---|---|
| Local dev | `http://localhost:8000` |
| Production | `https://api.visiondx.rw` |

Set in your Next.js `.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## Authentication Flow

### Register
```typescript
// POST /api/v1/auth/register
const res = await fetch(`${API_URL}/api/v1/auth/register`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    email: "user@example.com",
    full_name: "Alice Lab",
    password: "SecurePass1",
    role: "lab_technician",   // admin | lab_technician | doctor
  }),
});
// 201 → { success: true, data: { id, email, full_name, role, ... } }
```

### Login
```typescript
// POST /api/v1/auth/login
const res = await fetch(`${API_URL}/api/v1/auth/login`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ email, password }),
});
const { data } = await res.json();
// data = { access_token, refresh_token, token_type: "bearer", expires_in: 1800 }

// Store tokens (httpOnly cookie preferred in production)
localStorage.setItem("access_token", data.access_token);
```

### Protected requests
```typescript
const headers = {
  Authorization: `Bearer ${localStorage.getItem("access_token")}`,
};
```

### Get current user
```typescript
// GET /api/v1/user/profile   (or /api/v1/auth/me)
const res = await fetch(`${API_URL}/api/v1/user/profile`, { headers });
// { id, email, full_name, role, facility_name, is_active, created_at }
```

---

## AI Prediction Flow

This maps directly to the "Upload → Analyse → Results" flow in your UI.

### Upload image and get prediction
```typescript
// POST /api/v1/predictions/predict  (multipart/form-data)
const formData = new FormData();
formData.append("file", imageFile);                // File object
formData.append("disease_type", "malaria");        // see DiseaseType below

const res = await fetch(`${API_URL}/api/v1/predictions/predict`, {
  method: "POST",
  headers: { Authorization: `Bearer ${token}` },
  body: formData,
});

const { data } = await res.json();
// data = {
//   id: "uuid",
//   disease_type: "malaria",
//   status: "completed",
//   predicted_class: "Ring Stage",          ← show in result card
//   confidence_score: 0.9142,              ← confidence bar
//   severity_level: "mild",               ← colour badge
//   recommendation: "Initiate ACT...",    ← clinical guidance panel
//   model_version: "yolov9-mock-v1.0",
//   inference_time_ms: 87.3,
//   raw_output: { bounding_boxes: [...], class_probabilities: {...} },
// }
```

### Disease modules (`disease_type` values)

| Value | Dashboard Card |
|---|---|
| `malaria` | Malaria Blood Smear |
| `tuberculosis` | TB Detection |
| `pneumonia` | Pneumonia X-Ray |
| `diabetic_retinopathy` | Diabetic Retinopathy |
| `skin_lesion` | Skin Lesion |
| `general` | General AI Scan |

### Severity badge colours

| Value | Colour (Tailwind) |
|---|---|
| `negative` | `bg-green-100 text-green-800` |
| `mild` | `bg-yellow-100 text-yellow-800` |
| `moderate` | `bg-orange-100 text-orange-800` |
| `severe` | `bg-red-100 text-red-800` |

---

## History

```typescript
// GET /api/v1/predictions/history?page=1&page_size=10&disease_type=malaria
const res = await fetch(
  `${API_URL}/api/v1/predictions/history?page=1&page_size=10`,
  { headers: { Authorization: `Bearer ${token}` } },
);
const { data } = await res.json();
// data = { items: [...], total: 42, page: 1, page_size: 10, pages: 5 }
```

---

## Error Handling

All errors follow this shape:
```json
{ "success": false, "code": "NOT_FOUND", "message": "Prediction '...' not found" }
```

| HTTP Status | Code | Meaning |
|---|---|---|
| 401 | `AUTHENTICATION_ERROR` | Missing / expired JWT |
| 403 | `AUTHORIZATION_ERROR` | Insufficient role |
| 404 | `NOT_FOUND` | Resource not found |
| 409 | `DUPLICATE` | Email already exists |
| 422 | `IMAGE_VALIDATION_ERROR` | Bad image file |
| 502 | `INFERENCE_ERROR` | AI service error |

---

## Recommended Next.js API Client (`lib/api.ts`)

```typescript
const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

function getToken() {
  return typeof window !== "undefined"
    ? localStorage.getItem("access_token")
    : null;
}

async function apiFetch(path: string, init: RequestInit = {}) {
  const token = getToken();
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      ...(init.headers as Record<string, string>),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.message ?? `HTTP ${res.status}`);
  }
  return res.json();
}

export const api = {
  auth: {
    register: (body: object) =>
      apiFetch("/api/v1/auth/register", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
    login: (body: object) =>
      apiFetch("/api/v1/auth/login", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
    me: () => apiFetch("/api/v1/user/profile"),
  },
  predictions: {
    predict: (formData: FormData) =>
      apiFetch("/api/v1/predictions/predict", { method: "POST", body: formData }),
    history: (params?: Record<string, string>) => {
      const qs = new URLSearchParams(params).toString();
      return apiFetch(`/api/v1/predictions/history${qs ? "?" + qs : ""}`);
    },
    get: (id: string) => apiFetch(`/api/v1/predictions/${id}`),
  },
};
```
