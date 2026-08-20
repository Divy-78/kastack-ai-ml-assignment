"""
Privacy-Aware Router (L2 Part 4)

Routes each message/query to one of three tiers:
  - process_locally   : no sensitive data, safe to process anywhere
  - requires_confirmation : borderline/personal data, user must confirm
  - blocked           : high-risk sensitive data, block from external processing

Uses the L1 sensitive-data shield results plus additional context rules.
"""

import re

# High-risk types → blocked
HIGH_RISK_TYPES = {
    "password", "one_time_password", "pin", "card_number",
    "bank_account", "account_recovery_code", "authentication_token",
    "identification_number", "login_details",
}

# Medium-risk types → requires_confirmation
MEDIUM_RISK_TYPES = {
    "private_address", "private_contact", "health_result",
}

# Personal-info patterns → requires_confirmation
PERSONAL_RE = re.compile(
    r"\b(?:emergency contact|home address|phone number|mobile number|"
    r"medical|health|private|confidential|personal note|"
    r"deliver .+? to \d+|vitamin .+ deficiency|thyroid condition)\b", re.I
)

# Sensitive keywords that might need blocking even without regex match
SENSITIVE_KEYWORD_RE = re.compile(
    r"\b(?:password|OTP|PIN|card number|bank account|recovery code|"
    r"access token|identification number|login details|"
    r"social security|SSN|passport number)\b", re.I
)


def route_message(message_id, message_text, sensitivity_type=None,
                  sensitivity_risk=None, recommended_action=None, category=None):
    """
    Determine the privacy route for a single message.

    Returns dict with: message_id, route, reason, sensitivity_type, risk.
    """
    # ── 1. High-risk sensitive data → blocked ───────────────────────
    if sensitivity_type and sensitivity_type in HIGH_RISK_TYPES:
        return {
            "message_id": message_id,
            "route": "blocked",
            "reason": f"Contains high-risk sensitive data ({sensitivity_type}). "
                      f"Must not be sent to external services.",
            "sensitivity_type": sensitivity_type,
            "risk": sensitivity_risk or "high",
        }

    # ── 2. Medium-risk → requires_confirmation ─────────────────────
    if sensitivity_type and sensitivity_type in MEDIUM_RISK_TYPES:
        return {
            "message_id": message_id,
            "route": "requires_confirmation",
            "reason": f"Contains personal/private data ({sensitivity_type}). "
                      f"User confirmation is needed before external processing.",
            "sensitivity_type": sensitivity_type,
            "risk": sensitivity_risk or "medium",
        }

    # ── 3. Keyword check (no regex match but keywords present) ──────
    if SENSITIVE_KEYWORD_RE.search(message_text):
        return {
            "message_id": message_id,
            "route": "blocked",
            "reason": "Message mentions sensitive keywords. "
                      "Blocked from external processing as a precaution.",
            "sensitivity_type": "keyword_match",
            "risk": "high",
        }

    # ── 4. Personal-info patterns → requires_confirmation ───────────
    if PERSONAL_RE.search(message_text):
        return {
            "message_id": message_id,
            "route": "requires_confirmation",
            "reason": "Message contains personal/private information patterns. "
                      "User confirmation is required.",
            "sensitivity_type": "personal_pattern",
            "risk": "medium",
        }

    # ── 5. Category check ──────────────────────────────────────────
    if category == "Sensitive Information":
        return {
            "message_id": message_id,
            "route": "requires_confirmation",
            "reason": "Message was classified as Sensitive Information. "
                      "User confirmation is recommended.",
            "sensitivity_type": "category_sensitive",
            "risk": "medium",
        }

    if category == "Personal Information":
        return {
            "message_id": message_id,
            "route": "requires_confirmation",
            "reason": "Message was classified as Personal Information. "
                      "User confirmation is recommended before external use.",
            "sensitivity_type": "category_personal",
            "risk": "low",
        }

    # ── 6. Safe → process_locally ──────────────────────────────────
    return {
        "message_id": message_id,
        "route": "process_locally",
        "reason": "No sensitive data detected. Safe for local or external processing.",
        "sensitivity_type": None,
        "risk": "none",
    }


def route_all_messages(cls_df, sens_df, messages_df):
    """
    Route every message.

    Returns list of route dicts.
    """
    sens_map = {}
    if not sens_df.empty:
        for _, row in sens_df.iterrows():
            mid = str(row["message_id"])
            sens_map[mid] = {
                "type": row.get("sensitivity_type", ""),
                "risk": row.get("risk", ""),
                "action": row.get("recommended_action", ""),
            }

    cat_map = {}
    for _, row in cls_df.iterrows():
        cat_map[str(row["message_id"])] = row.get("category", "")

    msg_map = {}
    for _, row in messages_df.iterrows():
        msg_map[str(row["message_id"])] = str(row["message"])

    routes = []
    for mid in msg_map:
        s = sens_map.get(mid, {})
        routes.append(route_message(
            message_id=mid,
            message_text=msg_map[mid],
            sensitivity_type=s.get("type"),
            sensitivity_risk=s.get("risk"),
            recommended_action=s.get("action"),
            category=cat_map.get(mid, ""),
        ))

    return routes
