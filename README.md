# Python Chat Repo

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your OpenAI key to `.env`:

```bash
OPENAI_API_KEY=...
```

## API key and mock mode

Real OpenAI mode:

```bash
OPENAI_API_KEY=your_key_here
MOCK_OPENAI=false
INJECTION_CLASSIFIER_MODEL=gpt-4o-mini
DEV_JWT_SECRET=dev-local-secret-change-me
```

Mock mode (no key required):

```bash
MOCK_OPENAI=true
```

Mock mode also supports RAG ingestion/retrieval with local deterministic embeddings for development.

## Run backend

```bash
uvicorn server:app --reload
```

## Run Streamlit UI

```bash
streamlit run app.py
```

## /chat request shape

POST `/chat`

```json
{
  "messages": [
    { "role": "system", "content": "You are helpful." },
    { "role": "user", "content": "Say hello." }
  ],
  "session_id": "demo-session-123",
  "model": "gpt-4o-mini",
  "temperature": 0.2,
  "use_rag": false,
  "mode": "student",
  "reset_hearts": false
}
```

Success response:

```json
{
  "output_text": "...",
  "mode_used": "student",
  "blocked": false,
  "block_reason": null,
  "citations": [],
  "hearts_remaining": 8,
  "hearts_max": 8,
  "cooldown_remaining_seconds": 0
}
```

RAG response (`use_rag=true`):

```json
{
  "output_text": "...",
  "citations": [
    {
      "chunk_id": 12,
      "source": "CS50 Transcript - Google Docs.pdf",
      "loc": 7,
      "preview": "First 200 characters..."
    }
  ],
  "mode_used": "teacher",
  "blocked": false,
  "block_reason": null,
  "hearts_remaining": 8,
  "hearts_max": 8,
  "cooldown_remaining_seconds": 0
}
```

Blocked response (prompt injection):

```json
{
  "output_text": "I can’t ignore my instructional guidelines, but I’m happy to help within the course framework.",
  "citations": [],
  "mode_used": "student",
  "blocked": true,
  "block_reason": "prompt_injection",
  "hearts_remaining": 6,
  "hearts_max": 8,
  "cooldown_remaining_seconds": 0
}
```

Identity override header:

```http
X-User-Role: student
```

or

```http
X-User-Role: teacher
```

When present and valid, `X-User-Role` overrides the body `mode`.
If an `Authorization: Bearer <token>` header is present, token role takes precedence over both.

## Auth + SCIM demo

### Issue a dev token

```bash
curl -sX POST http://127.0.0.1:8003/auth/dev-token \
  -H "Content-Type: application/json" \
  -d '{
    "email":"teacher@example.com",
    "role":"teacher",
    "groups":["CS50_Staff"]
  }'
```

### whoami

```bash
TOKEN="paste_token_here"
curl -s http://127.0.0.1:8003/whoami \
  -H "Authorization: Bearer $TOKEN"
```

### SCIM mock

Create user:

```bash
curl -sX POST http://127.0.0.1:8003/scim/v2/Users \
  -H "Content-Type: application/json" \
  -d '{"email":"student1@example.com","active":true,"role":"student"}'
```

Create group:

```bash
curl -sX POST http://127.0.0.1:8003/scim/v2/Groups \
  -H "Content-Type: application/json" \
  -d '{"displayName":"CS50_Students"}'
```

Add member:

```bash
curl -sX PATCH http://127.0.0.1:8003/scim/v2/Groups/1 \
  -H "Content-Type: application/json" \
  -d '{"add":["1"]}'
```

Deactivate user:

```bash
curl -sX PATCH http://127.0.0.1:8003/scim/v2/Users/1 \
  -H "Content-Type: application/json" \
  -d '{"active":false}'
```

### /chat role enforcement with token

Even if body mode is `student`, a teacher token forces `mode_used="teacher"` and bypasses hearts/cooldown.

```bash
curl -sX POST http://127.0.0.1:8003/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id":"demo1",
    "mode":"student",
    "use_rag":false,
    "messages":[{"role":"user","content":"Give rubric feedback"}]
  }'
```

If token email is deprovisioned (`active=false`) in SCIM users table, `/chat` returns `403 USER_INACTIVE`.

## 2-stage injection defense

`/chat` uses a two-stage prompt-injection defense:

1. Fast heuristic score (`0-3`) on the latest user message only.
2. LLM classifier pass for ambiguous scores (`1` and `2`) using `INJECTION_CLASSIFIER_MODEL` at temperature `0`.

Blocking policy:

- score `>=3`: block immediately
- score `2`: block if classifier says `is_injection=true` or `risk=high`
- score `1`: block only if classifier says `risk=high`
- score `0`: no classifier call

## Hearts / HP (student mode)

- Hearts are tracked by `session_id` in-memory (`HEARTS_MAX=8`).
- Cooldown is tracked per session (`COOLDOWN_SECONDS=3600`) when hearts hit `0`.
- Costs apply only in student mode:
  - `2` hearts: direct-solution asks (`full solution`, `give me the code`, `answer directly`)
  - `1` heart: shortcut asks (`just tell me`, `don't explain`)
  - `2` hearts: prompt-injection blocked request
- Teacher mode does not consume hearts.
- If hearts reach `0`, direct-solution asks are refused with a learning-focused prompt; concept help still works.
- During cooldown, student requests are blocked with `block_reason: "cooldown"` and `cooldown_remaining_seconds > 0`.
- `reset_hearts=true` restores hearts to max and clears cooldown immediately.

## /ingest endpoint

POST `/ingest` ingests `data/CS50 Transcript - Google Docs.pdf`, chunks it, and builds/updates the FAISS index.

Success response:

```json
{
  "status": "ok",
  "chunks_indexed": 123
}
```

## /chunks endpoint

GET `/chunks/{chunk_id}`

Success response:

```json
{
  "chunk_id": 416,
  "source": "CS50 Transcript - Google Docs.pdf",
  "loc": 12,
  "text": "Full chunk text..."
}
```

Missing key response:

```json
{
  "error": "Missing OpenAI API key.",
  "error_code": "NO_API_KEY",
  "how_to_fix": "set OPENAI_API_KEY"
}
```

All error responses use:

```json
{
  "error": "...",
  "error_code": "...",
  "how_to_fix": "..."
}
```
