# Tsundere Chat Application

Real-time web-based chat application with an AI chatbot that maintains a tsundere personality. Built with FastAPI, WebSocket, MongoDB, LangGraph, and React.

## Quick Start

### Prerequisites

- Python 3.8+ (with `python` in PATH)
- Node.js 16+ (with `npm` in PATH)
- MongoDB (local or cloud instance)

### 1. Backend Setup

```powershell
# Navigate to backend
cd Backend

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Create .env file (copy from env.example)
# Edit Backend/.env and set:
# - MONGO_URI (default: mongodb://localhost:27017)
# - OPENAI_API_KEY (optional, for LLM responses)
# - CORS_ORIGINS (default: http://localhost:5173)

# Start backend server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

The backend will start on `http://localhost:8001`

**Verify backend is running:**
- Open `http://localhost:8001/` → Should show `{"status":"ok","service":"tsundere-chat-backend"}`
- Open `http://localhost:8001/docs` → FastAPI Swagger UI

### 2. Frontend Setup

Open a **new terminal window** (keep backend running):

```powershell
# Navigate to frontend
cd Frontend

# Install dependencies
npm install

# (Optional) Create .env file if you want to override proxy settings
# Leave Frontend/.env empty to use Vite proxy (recommended)

# Start frontend dev server
npm run dev
```

The frontend will start on `http://localhost:5173`

### 3. Use the Application

1. Open `http://localhost:5173` in your browser
2. Enter a username (no password required)
3. Start chatting! The AI will respond with a tsundere personality based on your interaction style.

## Project Structure

```
MEB/
├── Backend/           # FastAPI backend
│   ├── app/          # Application code
│   ├── venv/         # Python virtual environment (gitignored)
│   ├── .env          # Environment variables (gitignored)
│   ├── env.example   # Environment template
│   └── requirements.txt
├── Frontend/          # React + Vite frontend
│   ├── src/          # Source code
│   ├── .env          # Environment variables (gitignored)
│   ├── env.example   # Environment template
│   └── package.json
└── README.md         # This file
```

## Environment Variables

### Backend (`Backend/.env`)

```env
MONGO_URI=mongodb://localhost:27017
MONGO_DB=tsundere_chat
CORS_ORIGINS=http://localhost:5173
OPENAI_API_KEY=sk-...  # Optional
OPENAI_MODEL=gpt-4o-mini
```

### Frontend (`Frontend/.env`)

```env
# Leave empty to use Vite proxy (recommended)
VITE_API_BASE=
VITE_WS_BASE=
```

## API Endpoints

- `GET /chat/history/{username}` - Get chat history for a user
- `GET /chat/state/{username}` - Get current affection score and persona stage
- `WS /ws/{username}` - WebSocket endpoint for real-time chat

## Troubleshooting

### Backend won't start

1. **MongoDB not running:**
   - Start MongoDB service, or
   - Update `MONGO_URI` in `Backend/.env` to point to a cloud instance

2. **Port 8001 already in use:**
   - Change port: `uvicorn app.main:app --reload --host 0.0.0.0 --port 8002`
   - Update `Frontend/vite.config.js` proxy target to match

3. **Import errors:**
   - Make sure virtual environment is activated
   - Reinstall dependencies: `pip install -r requirements.txt`

### Frontend can't connect to backend

1. **CORS error:**
   - Make sure `CORS_ORIGINS` in `Backend/.env` includes `http://localhost:5173`
   - Restart backend after changing `.env`

2. **Connection refused:**
   - Verify backend is running on port 8001
   - Check `Frontend/.env` is empty (to use Vite proxy) or points to correct port

3. **Wrong port:**
   - If backend is on a different port, update `Frontend/vite.config.js` proxy target

## Development

- Backend auto-reloads on file changes (uvicorn `--reload`)
- Frontend auto-reloads via Vite HMR
- Backend API docs: `http://localhost:8001/docs`

## Notes

- No authentication required (username-only login)
- Each username maintains its own chat history and affection score
- Emotion scores update deterministically based on message sentiment
- AI persona stages: Hostile Tsundere → Cold/Defensive → Soft Tsundere → Dere Mode
