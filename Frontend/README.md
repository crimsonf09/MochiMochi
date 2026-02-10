# Tsundere Chat Frontend (React + Vite)

Dark-theme chat UI for the tsundere AI demo.

## Run

From `Frontend/`:

```powershell
npm install
npm run dev
```

Then open Vite’s dev URL (usually `http://localhost:5173`).

## Backend connectivity

By default, `vite.config.js` proxies:

- `/chat/*` → `http://localhost:8001`
- `/ws/*` → `ws://localhost:8001`

Optionally set env vars (see `env.example`):

- `VITE_API_BASE`
- `VITE_WS_BASE`
