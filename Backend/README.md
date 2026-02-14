# Tsundere Chat Backend (FastAPI + WebSocket + MongoDB)

## Quick Start

### 1. Setup

```powershell
# Navigate to Backend directory
cd Backend

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file (copy from `env.example`):

```env
MONGO_URI=mongodb://localhost:27017
MONGO_DB=tsundere_chat
CORS_ORIGINS=http://localhost:5173
OPENAI_API_KEY=sk-...  # Optional: enables LLM responses
OPENAI_MODEL=gpt-4o-mini
```

**Environment Variables:**
- `MONGO_URI` - MongoDB connection string (default: `mongodb://localhost:27017`)
- `MONGO_DB` - Database name (default: `tsundere_chat`)
- `CORS_ORIGINS` - Allowed frontend origins, comma-separated (default: `http://localhost:5173`)
- `OPENAI_API_KEY` - Optional; if unset, uses deterministic fallback responder
- `OPENAI_MODEL` - OpenAI model to use (default: `gpt-4o-mini`)

### 3. Run Server

```powershell
# Make sure virtual environment is activated
.\venv\Scripts\Activate.ps1

# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

**Expected output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
INFO:     Application startup complete.
[OK] Backend started: MongoDB connected, CORS origins: ['http://localhost:5173']
```

**Verify it's working:**
- Health check: `http://localhost:8001/` → `{"status":"ok","service":"tsundere-chat-backend"}`
- API docs: `http://localhost:8001/docs` → FastAPI Swagger UI

## API Endpoints

- `GET /chat/history/{username}` - Get full chat history for a user
- `GET /chat/state/{username}` - Get current affection score and persona stage
- `WS /ws/{username}` - WebSocket endpoint for real-time bidirectional chat

## Troubleshooting

### MongoDB Connection Error

**Error:** `[ERROR] Startup error: ...`

**Solutions:**
1. **Local MongoDB:** Make sure MongoDB service is running
   ```powershell
   # Check if MongoDB is running (Windows)
   Get-Service MongoDB
   ```

2. **Cloud MongoDB (Atlas):** Update `MONGO_URI` in `.env`:
   ```env
   MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/tsundere_chat
   ```

3. **Test MongoDB connection:**
   ```powershell
   # Using MongoDB shell (if installed)
   mongosh mongodb://localhost:27017
   ```

### Port Already in Use

**Error:** `Address already in use`

**Solution:** Use a different port:
```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8002
```
Then update `Frontend/vite.config.js` proxy target to match.

### Import Errors

**Error:** `ModuleNotFoundError` or `ImportError`

**Solution:**
```powershell
# Make sure venv is activated
.\venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt
```

## Architecture

- **FastAPI** - REST API and WebSocket server
- **MongoDB** - Persistent storage for chat history and emotion state
- **LangGraph** - Deterministic emotion scoring and response generation pipeline
- **Motor** - Async MongoDB driver
- **WebSocket** - Real-time bidirectional communication

## Development

- Server auto-reloads on code changes (`--reload` flag)
- API documentation available at `/docs` endpoint
- CORS configured for frontend development

