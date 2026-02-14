# GPT Error Troubleshooting Guide

## Common GPT Errors and Solutions

### 1. Authentication Error

**Error Message:**
```
[LangGraph] ✗ GPT Authentication Error: ...
→ Check your OPENAI_API_KEY in .env file
```

**Causes:**
- Invalid API key
- API key not set
- API key expired or revoked

**Solutions:**
1. **Check your `.env` file:**
   ```env
   OPENAI_API_KEY=sk-your-actual-key-here
   ```

2. **Verify API key format:**
   - Should start with `sk-`
   - Should be your full API key from OpenAI dashboard

3. **Get a new API key:**
   - Go to https://platform.openai.com/api-keys
   - Create a new API key
   - Copy it to your `.env` file

4. **Restart the backend** after updating `.env`

### 2. Rate Limit Error

**Error Message:**
```
[LangGraph] ✗ GPT Rate Limit Error: ...
→ You've exceeded your API rate limit
```

**Causes:**
- Too many requests in a short time
- Free tier limits exceeded
- Billing limit reached

**Solutions:**
1. **Wait a few minutes** and try again
2. **Check your OpenAI usage:** https://platform.openai.com/usage
3. **Upgrade your plan** if needed
4. **Reduce request frequency** (add delays between messages)

### 3. Model Not Found Error

**Error Message:**
```
[LangGraph] ✗ GPT Error (InvalidRequestError): Model not found
```

**Causes:**
- Invalid model name
- Model not available in your account
- Typo in model name

**Solutions:**
1. **Check your model name in `.env`:**
   ```env
   OPENAI_MODEL=gpt-4o-mini  # Correct
   OPENAI_MODEL=gpt-4-mini   # Wrong (typo)
   ```

2. **Common valid models:**
   - `gpt-4o-mini` (recommended, cheapest)
   - `gpt-4o`
   - `gpt-4-turbo`
   - `gpt-3.5-turbo`

3. **Verify model availability:**
   - Check https://platform.openai.com/docs/models
   - Some models require specific API access

### 4. Network/Connection Error

**Error Message:**
```
[LangGraph] ✗ GPT Error (ConnectionError): ...
→ Possible causes:
  - Network connection issue
```

**Causes:**
- No internet connection
- Firewall blocking OpenAI API
- Proxy/VPN issues
- OpenAI API downtime

**Solutions:**
1. **Check internet connection**
2. **Test OpenAI API directly:**
   ```powershell
   curl https://api.openai.com/v1/models -H "Authorization: Bearer sk-your-key"
   ```

3. **Check firewall settings**
4. **Check OpenAI status:** https://status.openai.com/

### 5. Invalid Request Error

**Error Message:**
```
[LangGraph] ✗ GPT API Error: Invalid request
```

**Causes:**
- Messages format incorrect
- Token limit exceeded
- Invalid parameters

**Solutions:**
1. **Check message format** (should be handled automatically)
2. **Reduce message history** (currently limited to 10 messages)
3. **Check token limits** (max_tokens is set to 500)

### 6. API Key Format Warning

**Warning Message:**
```
[WARNING] API key format looks invalid (should start with 'sk-')
```

**Solution:**
- Make sure your API key starts with `sk-`
- Check for extra spaces or quotes in `.env` file
- Example: `OPENAI_API_KEY=sk-proj-...` (not `"sk-proj-..."`)

## Debugging Steps

### Step 1: Verify Configuration

Check your `Backend/.env` file:
```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
```

### Step 2: Test API Key

Test your API key directly:
```powershell
# Install curl if needed, or use PowerShell
$headers = @{
    "Authorization" = "Bearer sk-your-key-here"
}
Invoke-RestMethod -Uri "https://api.openai.com/v1/models" -Headers $headers
```

### Step 3: Check Backend Logs

When you send a message, watch the backend console for:
```
[LangGraph] ✓ GPT response generated (stage: ..., affection: ...)
```

Or if there's an error:
```
[LangGraph] ✗ GPT Error: ...
```

### Step 4: Verify Model Access

1. Go to https://platform.openai.com/account/limits
2. Check if your model is available
3. Verify you have API access

## Fallback Behavior

**Important:** If GPT fails for any reason, the system automatically falls back to a deterministic responder. You'll see:

```
[LangGraph] Using fallback responder (stage: ..., affection: ...)
```

The chat will continue working, but responses will be template-based instead of GPT-generated.

## Testing GPT Integration

1. **Start backend with API key set:**
   ```powershell
   cd Backend
   .\venv\Scripts\Activate.ps1
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
   ```

2. **Look for startup message:**
   ```
   [OK] GPT Integration: ENABLED
   [OK] GPT Model: gpt-4o-mini
   ```

3. **Send a test message** and check console:
   - Success: `[LangGraph] ✓ GPT response generated`
   - Error: `[LangGraph] ✗ GPT Error: ...`

## Still Having Issues?

1. **Check OpenAI account:**
   - https://platform.openai.com/account
   - Verify billing is set up
   - Check usage limits

2. **Review error logs:**
   - Full error message in backend console
   - Error type (AuthenticationError, RateLimitError, etc.)

3. **Test with minimal example:**
   ```python
   from openai import OpenAI
   client = OpenAI(api_key="sk-your-key")
   response = client.chat.completions.create(
       model="gpt-4o-mini",
       messages=[{"role": "user", "content": "Hello"}]
   )
   print(response.choices[0].message.content)
   ```

4. **Contact OpenAI support** if the issue persists
