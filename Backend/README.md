# Tsundere Chat Backend (FastAPI + WebSocket + MongoDB)

## Endpoints

- `GET /chat/history/{username}`: full persisted chat history
- `GET /chat/state/{username}`: current affection score + persona stage
- `WS /ws/{username}`: real-time chat (send user messages, receive AI messages)

## Environment

Copy `env.example` to a local env file (or set env vars in your shell):

- `MONGO_URI` (default `mongodb://localhost:27017`)
- `MONGO_DB` (default `tsundere_chat`)
- `CORS_ORIGINS` (default `http://localhost:5173`)
- `OPENAI_API_KEY` (optional; if unset the backend uses a deterministic fallback responder)
- `OPENAI_MODEL` (default `gpt-4o-mini`)

## Run (Windows / PowerShell)

From `Backend/`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

Make sure MongoDB is running locally (or point `MONGO_URI` at your cluster).

