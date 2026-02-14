# How to Fix "Rate Limit Exceeded" Error

## Quick Fixes

### 1. **Wait and Retry (Automatic)**
The code now automatically retries with exponential backoff:
- **Attempt 1:** Immediate
- **Attempt 2:** Wait 2 seconds, retry
- **Attempt 3:** Wait 4 seconds, retry
- **Attempt 4:** Wait 8 seconds, retry

If all retries fail, it falls back to the deterministic responder.

### 2. **Check Your Usage**
1. Go to: https://platform.openai.com/usage
2. Check your current usage and limits
3. See if you've hit:
   - **RPM (Requests Per Minute)** limit
   - **TPM (Tokens Per Minute)** limit
   - **Daily/monthly spending** limit

### 3. **Wait a Few Minutes**
Rate limits reset over time:
- **RPM limits:** Reset every minute
- **TPM limits:** Reset every minute
- **Daily limits:** Reset at midnight UTC

**Solution:** Wait 1-2 minutes and try again.

## Long-Term Solutions

### Option 1: Upgrade Your OpenAI Plan

**Free Tier Limits:**
- Very low RPM/TPM limits
- Limited requests per day

**Paid Tier Benefits:**
- Higher rate limits
- More tokens per minute
- Better reliability

**How to upgrade:**
1. Go to: https://platform.openai.com/account/billing
2. Add payment method
3. Choose a plan or pay-as-you-go
4. Limits increase automatically

### Option 2: Reduce Request Frequency

**Add delays between messages:**
- The code already has retry logic
- You can add client-side rate limiting

**Example:** Limit to 1 request per 2 seconds:
```python
# In your frontend, add a small delay between messages
await new Promise(resolve => setTimeout(resolve, 2000)); // 2 second delay
```

### Option 3: Use a Cheaper Model

**Current:** `gpt-4o-mini` (recommended, already cheapest)

**If still hitting limits:**
- Consider `gpt-3.5-turbo` (even cheaper, but less capable)
- Or reduce `max_tokens` (currently 5000)

### Option 4: Implement Request Queuing

For high-traffic scenarios, implement a request queue:
- Queue requests instead of sending immediately
- Process one at a time with delays
- Prevents hitting rate limits

## Understanding Rate Limits

### Types of Limits

1. **RPM (Requests Per Minute)**
   - How many API calls you can make per minute
   - Example: 3 RPM = 3 requests per minute max

2. **TPM (Tokens Per Minute)**
   - Total tokens (input + output) per minute
   - Example: 40,000 TPM = 40k tokens per minute max

3. **Daily/Monthly Limits**
   - Spending limits
   - Usage caps

### Check Your Current Limits

1. Go to: https://platform.openai.com/account/limits
2. See your current tier limits
3. Check usage vs. limits

## Code Improvements Already Added

✅ **Automatic Retry Logic**
- Retries up to 3 times
- Exponential backoff (2s, 4s, 8s)
- Handles rate limit errors gracefully

✅ **Fallback Responder**
- If GPT fails, uses deterministic responder
- Chat continues working
- No user-facing errors

✅ **Better Error Messages**
- Shows retry attempts
- Suggests wait times
- Links to usage dashboard

## Immediate Actions

### If You're Getting Rate Limit Errors:

1. **Check your usage:**
   ```
   https://platform.openai.com/usage
   ```

2. **Wait 1-2 minutes** and try again
   - The code will auto-retry
   - Or manually wait and send another message

3. **Check your plan:**
   ```
   https://platform.openai.com/account/billing
   ```
   - Free tier has very low limits
   - Paid tier has much higher limits

4. **Upgrade if needed:**
   - Add payment method
   - Limits increase automatically

5. **Reduce request frequency:**
   - Don't send messages too quickly
   - Add small delays between messages

## Testing After Fix

1. **Restart backend:**
   ```powershell
   cd Backend
   .\venv\Scripts\Activate.ps1
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
   ```

2. **Send a test message**
   - Watch console for retry attempts
   - Should see: `[LangGraph] ✓ GPT response generated`

3. **If still failing:**
   - Check error message in console
   - Verify your OpenAI account status
   - Check billing/usage limits

## Prevention Tips

1. **Monitor usage regularly:**
   - Check https://platform.openai.com/usage daily
   - Set up usage alerts if available

2. **Use appropriate models:**
   - `gpt-4o-mini` is cheapest and recommended
   - Avoid `gpt-4` unless necessary (much more expensive)

3. **Optimize token usage:**
   - Current `max_tokens=5000` is high
   - Consider reducing if responses are shorter
   - Shorter responses = fewer tokens = less likely to hit TPM limits

4. **Implement caching:**
   - Cache similar responses
   - Reduce redundant API calls

## Emergency: If All Else Fails

The system automatically falls back to the deterministic responder:
- Chat continues working
- Responses are template-based (not GPT)
- No user-facing errors
- You can continue using the app

To re-enable GPT:
1. Wait for rate limits to reset (usually 1-60 minutes)
2. Upgrade your OpenAI plan
3. Reduce request frequency

## Summary

**Quick Fix:** Wait 1-2 minutes, the code will auto-retry

**Long-term Fix:** Upgrade your OpenAI plan for higher limits

**Current Status:** Code now handles rate limits gracefully with automatic retries and fallback
