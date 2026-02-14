# Tsundere Chat Frontend (React + Vite)

Dark-theme chat UI for the tsundere AI demo.

## Run

From `Frontend/`:

```powershell
npm install
npm run dev
```

Then open Vite’s dev URL (usually `http://localhost:5173`).

## Backend Setup (Required)

Before running the frontend, start the backend server:

```powershell
# In a separate terminal, navigate to Backend
cd Backend

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Start backend server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

**Verify backend is running:**
- Open `http://localhost:8001/` → Should show `{"status":"ok","service":"tsundere-chat-backend"}`
- Open `http://localhost:8001/docs` → FastAPI Swagger UI

## Backend Connectivity

By default, `vite.config.js` proxies requests to the backend:

- `/chat/*` → `http://localhost:8001`
- `/ws/*` → `ws://localhost:8001`

**To use the proxy (recommended):**
- Leave `Frontend/.env` empty or don't create it
- Requests will automatically go through the Vite proxy

**To bypass the proxy:**
Optionally set env vars in `Frontend/.env` (see `env.example`):

```env
VITE_API_BASE=http://localhost:8001
VITE_WS_BASE=ws://localhost:8001
```
