"""
Priority & Action Engine (L2 Part 1)

Assigns a priority (critical / high / medium / low) to every actionable
message using multiple signals.  Re-evaluates when later messages change
the deadline, urgency, or status of an existing task/event.
"""

import re
from datetime import datetime, timedelta

# ── signal detectors ────────────────────────────────────────────────

URGENT_RE = re.compile(
    r"\b(?:urgent|asap|critical|high[\s-]?priority|immediately|right away)\b", re.I
)
RESPONSE_RE = re.compile(
    r"\b(?:please confirm|any update|any progress|has .+ been handled|"
    r"is it in progress|still needs attention|please check)\b", re.I
)
COMPLETION_RE = re.compile(
    r"\b(?:completed successfully|has been completed|been submitted|"
    r"been finished|is done|finished successfully)\b", re.I
)
CANCEL_RE = re.compile(
    r"\b(?:cancel|no longer (?:needed|required)|not needed|cancelled)\b", re.I
)
RESCHEDULE_RE = re.compile(
    r"\b(?:moved to|rescheduled to|has been moved|new schedule|"
    r"extended to|pushed to|postponed to)\b", re.I
)
DEADLINE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
AUTHORITY_SENDERS = {"project lead", "mentor", "hr team", "manager", "director"}
AMBIGUOUS_RE = re.compile(
    r"\b(?:might|may|not completely sure|cannot confirm|"
    r"wait for confirmation|uncertain|could be)\b", re.I
)


def _days_until(deadline_str, ref_date):
    """Return days between ref_date and deadline. Negative = overdue."""
    try:
        dl = datetime.strptime(deadline_str, "%Y-%m-%d").date()
        return (dl - ref_date).days
    except (ValueError, TypeError):
        return None


def _deadline_signal(days):
    if days is None:
        return None, 0.0
    if days < 0:
        return "overdue", 1.0
    if days == 0:
        return "deadline_today", 0.95
    if days == 1:
        return "deadline_tomorrow", 0.85
    if days <= 3:
        return "deadline_within_3_days", 0.65
    if days <= 7:
        return "deadline_within_week", 0.40
    return "deadline_distant", 0.15


def compute_priority(message_text, deadline_str, ref_date, category,
                     sensitivity_type, sender, related_msgs=None,
                     status_override=None):
    """
    Return (priority, reason, signals, confidence).

    Parameters
    ----------
    message_text : str
    deadline_str : str or None     – "YYYY-MM-DD"
    ref_date     : date            – reference date (message timestamp date)
    category     : str             – L1 classification
    sensitivity_type : str or None – from sensitive shield
    sender       : str
    related_msgs : list[dict] or None – earlier messages in same group
    status_override : str or None  – "completed" / "cancelled" etc.
    """
    signals = []
    weights = []

    # ── 1. status override (completed / cancelled) ──────────────────
    if status_override in ("completed", "cancelled"):
        return (
            "low",
            f"The task/event has been {status_override}; priority is reduced.",
            [f"status_{status_override}"],
            0.96,
        )

    # ── 2. check for completion / cancellation in text ──────────────
    if COMPLETION_RE.search(message_text):
        return (
            "low",
            "Message confirms task completion.",
            ["completion_confirmed"],
            0.95,
        )
    if CANCEL_RE.search(message_text):
        return (
            "low",
            "Message cancels the task or marks it as no longer needed.",
            ["cancellation_confirmed"],
            0.95,
        )

    # ── 3. ambiguous / unclear ──────────────────────────────────────
    is_ambiguous = bool(AMBIGUOUS_RE.search(message_text))
    if is_ambiguous:
        signals.append("ambiguous_status")
        weights.append(0.10)

    # ── 4. deadline proximity ───────────────────────────────────────
    days = _days_until(deadline_str, ref_date) if deadline_str else None
    dl_signal, dl_weight = _deadline_signal(days)
    if dl_signal:
        signals.append(dl_signal)
        weights.append(dl_weight)

    # ── 5. urgency keywords ─────────────────────────────────────────
    if URGENT_RE.search(message_text):
        signals.append("urgent_keyword")
        weights.append(0.90)

    # ── 6. response required ────────────────────────────────────────
    if RESPONSE_RE.search(message_text):
        signals.append("response_required")
        weights.append(0.45)

    # ── 7. follow-up pressure ───────────────────────────────────────
    if related_msgs:
        followup_count = sum(
            1 for m in related_msgs
            if RESPONSE_RE.search(str(m.get("message", "")))
        )
        if followup_count >= 3:
            signals.append("high_follow_up_pressure")
            weights.append(0.70)
        elif followup_count >= 1:
            signals.append("follow_up_pressure")
            weights.append(0.40)

    # ── 8. sensitivity ──────────────────────────────────────────────
    if sensitivity_type:
        signals.append("contains_sensitive_data")
        weights.append(0.50)

    # ── 9. sender authority ─────────────────────────────────────────
    if sender and sender.strip().lower() in AUTHORITY_SENDERS:
        signals.append("authority_sender")
        weights.append(0.30)

    # ── 10. reschedule detected ─────────────────────────────────────
    if RESCHEDULE_RE.search(message_text):
        signals.append("rescheduled")
        weights.append(0.35)

    # ── 11. category-based baseline ─────────────────────────────────
    if category == "Action Required":
        weights.append(0.30)
    elif category == "Meeting or Event":
        weights.append(0.25)
    elif category == "Sensitive Information":
        weights.append(0.40)
    else:
        weights.append(0.10)

    # ── aggregate ───────────────────────────────────────────────────
    if not weights:
        score = 0.10
    else:
        score = max(weights)           # dominant signal
        avg   = sum(weights) / len(weights)
        # If multiple strong signals co-occur, use more aggressive blend
        strong = [w for w in weights if w >= 0.65]
        if len(strong) >= 2:
            score = 0.8 * score + 0.2 * avg  # strong co-occurrence
        else:
            score = 0.6 * score + 0.4 * avg  # blend

    if score >= 0.80:
        priority = "critical"
    elif score >= 0.55:
        priority = "high"
    elif score >= 0.30:
        priority = "medium"
    else:
        priority = "low"

    # confidence is higher when more signals agree
    confidence = min(0.99, round(0.60 + 0.08 * len(signals), 2))

    # reason
    if not signals:
        signals = ["baseline_category"]
    reason_parts = []
    if "overdue" in signals:
        reason_parts.append(f"the deadline ({deadline_str}) has passed")
    if "deadline_today" in signals:
        reason_parts.append(f"the deadline is today ({deadline_str})")
    if "deadline_tomorrow" in signals:
        reason_parts.append(f"the deadline is tomorrow ({deadline_str})")
    if "urgent_keyword" in signals:
        reason_parts.append("the message uses urgent language")
    if "high_follow_up_pressure" in signals:
        reason_parts.append("multiple follow-up messages have been sent")
    if "follow_up_pressure" in signals:
        reason_parts.append("a follow-up was sent")
    if "response_required" in signals:
        reason_parts.append("a response is requested")
    if "contains_sensitive_data" in signals:
        reason_parts.append("the message contains sensitive data")
    if "authority_sender" in signals:
        reason_parts.append(f"the sender ({sender}) is an authority figure")
    if "rescheduled" in signals:
        reason_parts.append("the schedule has been changed")
    if "ambiguous_status" in signals:
        reason_parts.append("the status is ambiguous or unconfirmed")
    if not reason_parts:
        reason_parts.append(f"category is {category}")
    reason = "Priority is {}: {}.".format(priority, "; ".join(reason_parts))

    return priority, reason, signals, confidence


def assign_priorities(cls_df, tasks_df, sens_df, groups=None):
    """
    Assign priority to every actionable message (Action Required,
    Meeting or Event, Sensitive Information).

    Returns a list of priority-record dicts.
    """
    actionable_cats = {"Action Required", "Meeting or Event", "Sensitive Information"}
    sens_map = {}
    if not sens_df.empty:
        for _, r in sens_df.iterrows():
            sens_map[str(r["message_id"])] = r.get("sensitivity_type", "")

    # build task lookup  message_id -> task row
    task_map = {}
    if not tasks_df.empty:
        for _, r in tasks_df.iterrows():
            task_map[str(r["source_message_id"])] = r

    # build group lookup  message_id -> group
    group_for_msg = {}
    group_msgs_cache = {}
    if groups:
        for g in groups:
            for mid in g.get("related_message_ids", []):
                group_for_msg[mid] = g

    results = []
    for _, row in cls_df.iterrows():
        cat = row["category"]
        if cat not in actionable_cats:
            continue

        mid = str(row["message_id"])
        text = str(row.get("message", row.get("reason", "")))
        sender = str(row.get("sender", ""))
        ts = row["timestamp"]
        if isinstance(ts, str):
            try:
                ref_date = datetime.strptime(ts[:10], "%Y-%m-%d").date()
            except ValueError:
                ref_date = datetime.now().date()
        else:
            ref_date = ts.date() if hasattr(ts, "date") else datetime.now().date()

        # deadline
        deadline = None
        task_row = task_map.get(mid)
        item_id = None
        if task_row is not None:
            deadline = task_row.get("date_or_deadline")
            item_id = task_row.get("item_id")
            if deadline and str(deadline).lower() in ("none", "nan", "nat", "unresolved"):
                deadline = None

        # status from group
        status_override = None
        grp = group_for_msg.get(mid)
        if grp:
            status_override = grp.get("status")
            if status_override not in ("completed", "cancelled"):
                status_override = None

        # related messages
        related = None
        if grp:
            related = [
                {"message": m_text}
                for m_text in grp.get("_message_texts", [])
            ]

        p, reason, sigs, conf = compute_priority(
            message_text=text,
            deadline_str=str(deadline) if deadline else None,
            ref_date=ref_date,
            category=cat,
            sensitivity_type=sens_map.get(mid),
            sender=sender,
            related_msgs=related,
            status_override=status_override,
        )
        results.append({
            "message_id": mid,
            "item_id": item_id or "",
            "priority": p,
            "reason": reason,
            "signals": sigs,
            "confidence": conf,
        })

    return results
