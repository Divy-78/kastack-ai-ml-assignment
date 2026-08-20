"""
Related-Message Grouping (L2 Part 2)

Groups messages that refer to the same task, meeting, event, request,
or subject using normalised task-title matching, TF-IDF cosine similarity,
and chronological ordering.
"""

import re
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── helpers ─────────────────────────────────────────────────────────

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
IN_PROGRESS_RE = re.compile(
    r"\b(?:in progress|working on|started|is it in progress)\b", re.I
)
AMBIGUOUS_RE = re.compile(
    r"\b(?:might|may|not completely sure|cannot confirm|"
    r"wait for confirmation|uncertain|could be)\b", re.I
)
DEADLINE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")

# Normalise a task/event title for matching
_STRIP_RE = re.compile(r"[^a-z0-9 ]+")

def _norm(title):
    if not title:
        return ""
    return _STRIP_RE.sub("", str(title).lower()).strip()


def _detect_status(texts):
    """Scan a list of message texts (chronologically) and return the latest status."""
    status = "pending"
    for t in texts:
        if COMPLETION_RE.search(t):
            status = "completed"
        elif CANCEL_RE.search(t):
            status = "cancelled"
        elif RESCHEDULE_RE.search(t):
            status = "rescheduled"
        elif IN_PROGRESS_RE.search(t):
            status = "in_progress"
        elif AMBIGUOUS_RE.search(t):
            status = "unclear"
    return status


def _latest_deadline(texts):
    """Return the chronologically last YYYY-MM-DD found across texts."""
    latest = None
    for t in texts:
        for m in DEADLINE_RE.finditer(t):
            latest = m.group(1)
    return latest


def _extract_subject(text):
    """Try to extract a task/event subject phrase from a message."""
    text_lower = text.lower()

    # Patterns like "following up on <subject>" or "update on <subject>"
    patterns = [
        r"(?:following up on|follow-up on|follow up on|any progress on the item concerning|"
        r"please check the latest status of|the work we discussed about|"
        r"has the .+? item been handled|i am referring to our earlier request about|"
        r"any update on|can you share an update on|"
        r"please confirm whether you started to|"
        r"update:\s*|you can cancel\s*|"
        r"the deadline to\s+|"
        r"please note that\s+)"
        r"(.+?)(?:\s*;\s*|\s*\.\s*|\s*\?\s*|\s*$)",
    ]
    for p in patterns:
        m = re.search(p, text_lower)
        if m:
            subj = m.group(1).strip()
            # Clean trailing junk
            subj = re.sub(r"\s+(?:is it in progress|is due on.*|has been completed.*|"
                          r"it is no longer.*|still needs.*|item been handled.*|"
                          r"is now.*|although.*|earlier than.*)", "", subj)
            subj = subj.strip(" .;?,!")
            if len(subj) > 3:
                return subj

    # Event rescheduling: "The <event> has been moved to..."
    m = re.search(r"the\s+(.+?)\s+has been moved to", text_lower)
    if m:
        return m.group(1).strip()

    # New task/event patterns
    m = re.search(r"new task:\s*(.+?)(?:\s+by\s+20\d{2}|$)", text_lower)
    if m:
        return m.group(1).strip()
    m = re.search(r"a new\s+(.+?)\s+(?:session\s+)?is scheduled", text_lower)
    if m:
        return m.group(1).strip()

    # "X might already be finished" / "X might already be done"
    m = re.search(r"^(.+?)\s+might\s+already\s+be", text_lower)
    if m:
        subj = m.group(1).strip()
        if len(subj) > 3:
            return subj

    # "this is another status request about X"
    m = re.search(r"(?:status request about|another.*request about)\s+(.+?)(?:\s*[.,;!?]|\s*$)", text_lower)
    if m:
        subj = m.group(1).strip()
        if len(subj) > 3:
            return subj

    # "the deadline for X has been extended to"
    m = re.search(r"the deadline for\s+(.+?)\s+has been\s+(?:extended|moved|pushed)", text_lower)
    if m:
        return m.group(1).strip()

    # "the <event> has been cancelled"
    m = re.search(r"the\s+(.+?)\s+has been\s+cancelled", text_lower)
    if m:
        return m.group(1).strip()

    return None


# ── main grouping function ──────────────────────────────────────────

def build_groups(cls_df, tasks_df, all_messages_df):
    """
    Group related messages.

    Parameters
    ----------
    cls_df          : DataFrame with classification results (message_id, category, …)
    tasks_df        : DataFrame with extracted tasks/events
    all_messages_df : Original messages DataFrame (message_id, timestamp, sender, message)

    Returns
    -------
    list[dict] – each dict is one group with fields per the spec.
    """

    # ── 1. Build a message lookup ───────────────────────────────────
    msg_lookup = {}
    for _, row in all_messages_df.iterrows():
        mid = str(row["message_id"])
        msg_lookup[mid] = {
            "message_id": mid,
            "timestamp": str(row["timestamp"]),
            "sender": str(row["sender"]),
            "message": str(row["message"]),
        }

    # ── 2. Attach category ──────────────────────────────────────────
    cat_map = {}
    for _, row in cls_df.iterrows():
        cat_map[str(row["message_id"])] = row["category"]

    # ── 3. Build task/event lookup ──────────────────────────────────
    task_for_msg = {}
    if not tasks_df.empty:
        for _, row in tasks_df.iterrows():
            task_for_msg[str(row["source_message_id"])] = {
                "item_id": row.get("item_id", ""),
                "title": str(row.get("title", "")),
                "type": row.get("type", "task"),
            }

    # ── 4. Extract subject for every message ────────────────────────
    subjects = {}
    for mid, info in msg_lookup.items():
        # If there's an extracted task/event title, use it
        t = task_for_msg.get(mid)
        if t and t["title"]:
            subjects[mid] = _norm(t["title"])
        else:
            subj = _extract_subject(info["message"])
            if subj:
                subjects[mid] = _norm(subj)

    # ── 5. Group by normalised subject ──────────────────────────────
    subj_groups = defaultdict(list)
    for mid, subj in subjects.items():
        if subj:
            subj_groups[subj].append(mid)

    # Also group messages that have no extracted subject but high TF-IDF
    # similarity to an existing group.
    ungrouped = [mid for mid in msg_lookup if mid not in subjects]

    # ── 6. TF-IDF fallback for ungrouped messages ───────────────────
    if ungrouped and subj_groups:
        # Build centroid for each group
        group_keys = list(subj_groups.keys())
        group_texts = []
        for gk in group_keys:
            combined = " ".join(msg_lookup[m]["message"] for m in subj_groups[gk] if m in msg_lookup)
            group_texts.append(combined)

        ungrouped_texts = [msg_lookup[m]["message"] for m in ungrouped if m in msg_lookup]
        valid_ungrouped = [m for m in ungrouped if m in msg_lookup]

        if ungrouped_texts and group_texts:
            all_texts = group_texts + ungrouped_texts
            try:
                vec = TfidfVectorizer(
                    lowercase=True, ngram_range=(1, 2),
                    max_features=8000, sublinear_tf=True
                )
                X = vec.fit_transform(all_texts)
                group_vecs = X[:len(group_texts)]
                ungrouped_vecs = X[len(group_texts):]
                sims = cosine_similarity(ungrouped_vecs, group_vecs)

                for i, mid in enumerate(valid_ungrouped):
                    best_idx = int(np.argmax(sims[i]))
                    if sims[i, best_idx] >= 0.35:
                        subj_groups[group_keys[best_idx]].append(mid)
            except Exception:
                pass  # If TF-IDF fails, skip fallback

    # ── 7. Build group output ───────────────────────────────────────
    groups = []
    gid = 1
    for subj, mids in subj_groups.items():
        if len(mids) < 2:
            continue  # single-message groups are not useful

        # Deduplicate and sort chronologically
        mids = sorted(set(mids), key=lambda m: msg_lookup.get(m, {}).get("timestamp", ""))

        texts = [msg_lookup[m]["message"] for m in mids if m in msg_lookup]
        status = _detect_status(texts)
        deadline = _latest_deadline(texts)

        # Related task/event IDs
        item_ids = []
        for m in mids:
            t = task_for_msg.get(m)
            if t and t["item_id"]:
                item_ids.append(t["item_id"])
        item_ids = sorted(set(item_ids))

        # Title: use the extracted task title or the subject key
        title = subj.replace("_", " ").title()
        for m in mids:
            t = task_for_msg.get(m)
            if t and t["title"]:
                title = t["title"]
                break

        # Summary
        n = len(mids)
        if status == "completed":
            summary = f"The task '{title}' was requested, followed up {n-1} time(s), and confirmed as completed."
        elif status == "cancelled":
            summary = f"The task '{title}' was requested, followed up {n-1} time(s), and then cancelled."
        elif status == "rescheduled":
            summary = f"The event '{title}' was scheduled, then rescheduled. Latest date: {deadline or 'unknown'}."
        elif status == "unclear":
            summary = f"The status of '{title}' is ambiguous across {n} messages."
        else:
            summary = f"The task/event '{title}' has {n} related messages and is currently {status}."

        confidence = round(min(0.98, 0.70 + 0.04 * min(n, 7)), 2)

        groups.append({
            "group_id": f"GROUP_{gid:03d}",
            "title": title,
            "related_message_ids": mids,
            "related_item_ids": item_ids,
            "status": status,
            "latest_deadline": deadline,
            "summary": summary,
            "confidence": confidence,
            "_message_texts": texts,  # internal, used by priority engine
        })
        gid += 1

    return groups
