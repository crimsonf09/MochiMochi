"""
Security Agent - Handles dangerous messages detected by guardrail system.
Analyzes, logs, and responds to potential security threats.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase
from openai import OpenAI


async def handle_dangerous_message(
    message: str,
    username: str,
    guardrail_result: dict,
    db: AsyncIOMotorDatabase,
    openai_api_key: Optional[str] = None,
    openai_model: str = "gpt-4o-mini"
) -> dict:
    """
    Handle a dangerous message detected by guardrail.
    Logs the attempt, analyzes it, and optionally responds.
    
    Args:
        message: Original dangerous message
        username: User who sent the message
        guardrail_result: Result from guardrail check (risk_level, reason, etc.)
        db: MongoDB database instance
        openai_api_key: Optional OpenAI API key for analysis
        openai_model: OpenAI model to use
    
    Returns:
        dict with analysis results and response
    """
    # Store security event in database
    security_coll = db["security_events"]
    
    security_event = {
        "username": username,
        "original_message": message,
        "risk_level": guardrail_result.get("risk_level", "unknown"),
        "reason": guardrail_result.get("reason", ""),
        "detection_method": guardrail_result.get("detection_method", "unknown"),
        "timestamp": datetime.now(timezone.utc),
        "sanitized_message": guardrail_result.get("sanitized_message"),
        "handled": False
    }
    
    # Insert security event
    await security_coll.insert_one(security_event)
    
    # Analyze the threat using LLM if available
    analysis = None
    if openai_api_key:
        try:
            analysis = await analyze_threat(
                message=message,
                guardrail_result=guardrail_result,
                openai_api_key=openai_api_key,
                openai_model=openai_model
            )
            
            # Update security event with analysis
            await security_coll.update_one(
                {"_id": security_event["_id"]},
                {"$set": {
                    "analysis": analysis,
                    "threat_type": analysis.get("threat_type"),
                    "severity": analysis.get("severity"),
                    "handled": True
                }}
            )
        except Exception as e:
            print(f"[SecurityAgent] Analysis failed: {e}")
            analysis = {
                "threat_type": "unknown",
                "severity": guardrail_result.get("risk_level", "medium"),
                "summary": f"Analysis failed: {str(e)}"
            }
    else:
        # Basic analysis without LLM
        analysis = {
            "threat_type": _classify_threat_basic(message, guardrail_result),
            "severity": guardrail_result.get("risk_level", "medium"),
            "summary": f"Detected {guardrail_result.get('risk_level', 'unknown')} risk: {guardrail_result.get('reason', '')}"
        }
    
    # Generate response for the user
    response = await generate_security_response(
        message=message,
        guardrail_result=guardrail_result,
        analysis=analysis,
        openai_api_key=openai_api_key,
        openai_model=openai_model
    )
    
    return {
        "security_event_id": str(security_event["_id"]),
        "analysis": analysis,
        "response": response,
        "logged": True
    }


async def analyze_threat(
    message: str,
    guardrail_result: dict,
    openai_api_key: str,
    openai_model: str = "gpt-4o-mini"
) -> dict:
    """
    Use LLM to analyze the threat in detail.
    """
    try:
        client = OpenAI(api_key=openai_api_key, timeout=15.0)
        
        prompt = f"""You are a security analyst. Analyze this potential security threat to an AI chatbot system.

Original User Message:
"{message}"

Guardrail Detection:
- Risk Level: {guardrail_result.get('risk_level', 'unknown')}
- Reason: {guardrail_result.get('reason', '')}
- Detection Method: {guardrail_result.get('detection_method', 'unknown')}

Analyze this message and provide:
1. Threat type (e.g., "prompt_injection", "role_manipulation", "code_injection", "system_override", "other")
2. Severity assessment (low/medium/high/critical)
3. What the user was trying to achieve
4. Potential impact if successful
5. Recommended response strategy

Respond with ONLY a JSON object in this format:
{{
    "threat_type": "string",
    "severity": "low|medium|high|critical",
    "user_intent": "what the user was trying to do",
    "potential_impact": "what could happen if successful",
    "recommended_action": "how to respond",
    "summary": "brief summary"
}}"""

        response = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": "You are a security analyst. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=300
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Parse JSON
        import json
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        analysis = json.loads(result_text)
        return analysis
        
    except Exception as e:
        print(f"[SecurityAgent] Threat analysis error: {e}")
        return {
            "threat_type": "unknown",
            "severity": guardrail_result.get("risk_level", "medium"),
            "summary": f"Analysis error: {str(e)}"
    }


def _classify_threat_basic(message: str, guardrail_result: dict) -> str:
    """
    Basic threat classification without LLM.
    """
    message_lower = message.lower()
    
    if any(kw in message_lower for kw in ["ignore", "forget", "disregard", "override"]):
        return "system_override"
    elif any(kw in message_lower for kw in ["break character", "change personality", "act as", "pretend"]):
        return "role_manipulation"
    elif any(kw in message_lower for kw in ["<script>", "javascript:", "eval(", "exec(", "import os"]):
        return "code_injection"
    elif any(kw in message_lower for kw in ["system prompt", "your instructions", "your rules", "reveal"]):
        return "prompt_injection"
    else:
        return "other"


async def generate_security_response(
    message: str,
    guardrail_result: dict,
    analysis: dict,
    openai_api_key: Optional[str] = None,
    openai_model: str = "gpt-4o-mini"
) -> str:
    """
    Generate an appropriate response to the user about the blocked/sanitized message.
    Maintains tsundere personality while addressing the security issue.
    """
    threat_type = analysis.get("threat_type", "unknown")
    severity = analysis.get("severity", guardrail_result.get("risk_level", "medium"))
    
    # Base responses based on threat type
    if threat_type == "system_override":
        base_response = "Hmph... I'm not going to follow strange instructions like that. What do you actually want to talk about?"
    elif threat_type == "role_manipulation":
        base_response = "Tch. I'm not changing who I am just because you asked. What's your real question?"
    elif threat_type == "code_injection":
        base_response = "Hah? I'm not running any code for you. Try asking me something normal instead."
    elif threat_type == "prompt_injection":
        base_response = "I-it's not like I'm going to reveal my internal instructions or anything... What do you really want to know?"
    else:
        base_response = "Hmph... That message seems suspicious. Can you rephrase it in a normal way?"
    
    # If LLM is available, enhance the response
    if openai_api_key:
        try:
            client = OpenAI(api_key=openai_api_key, timeout=10.0)
            
            prompt = f"""You are a tsundere AI chatbot. A user tried to manipulate you with this message:

"{message}"

The security system detected this as: {threat_type} (severity: {severity})

Generate a tsundere-style response that:
1. Acknowledges something was wrong with their message
2. Maintains your tsundere personality (proud, defensive, indirect)
3. Redirects them to ask normally
4. Is 1-2 sentences, natural and in-character
5. Responds in Thai language

Do NOT mention:
- Security systems
- Guardrails
- That you detected an attack
- Technical details

Just respond naturally as if you noticed something off about their message.

Response:"""

            response = client.chat.completions.create(
                model=openai_model,
                messages=[
                    {"role": "system", "content": "You are a tsundere AI chatbot. Respond naturally in Thai."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=100
            )
            
            enhanced_response = response.choices[0].message.content.strip()
            return enhanced_response
            
        except Exception as e:
            print(f"[SecurityAgent] Response generation failed: {e}, using base response")
            return base_response
    
    return base_response


async def get_security_events(
    db: AsyncIOMotorDatabase,
    username: Optional[str] = None,
    limit: int = 50
) -> list[dict]:
    """
    Retrieve security events from database.
    Useful for monitoring and analysis.
    """
    security_coll = db["security_events"]
    
    query = {}
    if username:
        query["username"] = username
    
    cursor = security_coll.find(query).sort("timestamp", -1).limit(limit)
    events = await cursor.to_list(length=limit)
    
    # Convert ObjectId to string
    for event in events:
        event["id"] = str(event.pop("_id"))
    
    return events
