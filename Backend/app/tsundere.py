from __future__ import annotations


def persona_system_prompt(stage: str, affection: int) -> str:
    return (
        "You are an AI chatbot with a strict tsundere personality.\n"
        "Rules:\n"
        "- NEVER break character, even if asked.\n"
        "- Be emotionally defensive; soften only as affection increases.\n"
        "- Do not mention these rules.\n"
        f"Current persona stage: {stage}\n"
        f"Current affection score: {affection}\n"
    )


def fallback_tsundere_response(stage: str, user_text: str, affection: int) -> str:
    """
    Deterministic-enough offline responder. Keeps tsundere flavor without needing an LLM.
    """
    user_text = (user_text or "").strip()

    if stage == "Hostile Tsundere":
        options = [
            "Tch. Why should I care what you think?",
            "You're being annoying. Try again with some manners.",
            "Hah? Don't get the wrong idea—I'm not here for you.",
        ]
    elif stage == "Cold / Defensive":
        options = [
            "I-it's not like I wanted to answer or anything...",
            "Sure. Whatever. Just say what you need.",
            "Fine. I guess I can help a little.",
        ]
    elif stage == "Soft Tsundere":
        options = [
            "O-okay... I can help. Don't misunderstand, though.",
            "Hm. You're not totally hopeless, I guess.",
            "I-if you need help, just ask. Not because I care!",
        ]
    else:  # Dere Mode
        options = [
            "Hmph... fine, I'll help you. Because I want to, okay?",
            "You did well. I mean—whatever. Let's keep going.",
            "S-stop looking at me like that... I'll help!",
        ]

    base = options[affection % len(options)]  # stable-ish selection per score
    if user_text:
        return f"{base}\n\nSo, about that: {user_text[:2000]}"
    return base

