"""
Message Classification for Memory System
Classifies messages as: fact, event, or regular conversation
"""

from __future__ import annotations

import re
from typing import Any, Literal


MessageType = Literal["fact", "event", "regular"]


def classify_message(message: str) -> tuple[MessageType, dict[str, Any]]:
    """
    Classify message type and extract information.
    Returns: (type, metadata)
    """
    message_lower = message.lower().strip()
    
    # Pattern matching for facts
    fact_patterns = [
        (r"my name is (\w+)", "name"),
        (r"i'm (\w+)", "name"),
        (r"i am (\w+)", "name"),
        (r"call me (\w+)", "name"),
        (r"my birthday is (.+)", "birthday"),
        (r"i was born on (.+)", "birthday"),
        (r"i'm (\d+) years old", "age"),
        (r"i work as (.+)", "job"),
        (r"i'm a (.+)", "job"),
        (r"my job is (.+)", "job"),
        (r"i like (.+)", "preference"),
        (r"i love (.+)", "preference"),
        (r"i hate (.+)", "preference"),
        (r"my goal is (.+)", "goal"),
        (r"i want to (.+)", "goal"),
    ]
    
    # Check for facts
    facts = {}
    for pattern, fact_type in fact_patterns:
        match = re.search(pattern, message_lower, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if fact_type == "name":
                facts["name"] = (value, 0.9)
            elif fact_type == "birthday":
                facts["birthday"] = (value, 0.8)
            elif fact_type == "age":
                facts["age"] = (value, 0.8)
            elif fact_type == "job":
                facts["job"] = (value, 0.8)
            elif fact_type == "preference":
                # Store as preference
                if "preferences" not in facts:
                    facts["preferences"] = []
                facts["preferences"].append(value)
            elif fact_type == "goal":
                facts["goal"] = (value, 0.7)
    
    if facts:
        return ("fact", {"facts": facts})
    
    # Check for significant events (emotional or meaningful)
    event_indicators = [
        r"thank you",
        r"i'm (?:really |very )?(?:happy|sad|angry|excited|disappointed|grateful)",
        r"that means (?:a lot|so much)",
        r"i (?:will|won't) (?:never|always) forget",
        r"this is (?:important|special|meaningful)",
        r"i (?:love|hate) (?:when|that)",
    ]
    
    for pattern in event_indicators:
        if re.search(pattern, message_lower):
            # Extract event summary (simplified)
            summary = message[:200]  # First 200 chars
            return ("event", {
                "summary": summary,
                "importance": 0.7,  # Default importance
            })
    
    # Default: regular conversation
    return ("regular", {})


def extract_facts_from_message(message: str) -> dict[str, tuple[str, float]]:
    """Extract structured facts from message"""
    msg_type, metadata = classify_message(message)
    
    if msg_type == "fact":
        facts = metadata.get("facts", {})
        result = {}
        
        for key, value in facts.items():
            if key == "preferences":
                # Handle list of preferences
                for pref in value:
                    result[f"preference_{len(result)}"] = (pref, 0.7)
            else:
                result[key] = value
        
        return result
    
    return {}
