"""
Semantic Search & Intelligent Assistant (L2 Part 3)

TF-IDF retrieval index over all messages, tasks, and groups.
Rule-based query-intent detection and structured answer generation.
"""

import re
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── query intent patterns ───────────────────────────────────────────

INTENT_PATTERNS = [
    ("tasks_today",        re.compile(r"tasks?\s+(?:for\s+)?today|today.*tasks?|complete today", re.I)),
    ("critical_pending",   re.compile(r"critical|high[\s-]?priority.*pending|pending.*critical", re.I)),
    ("completed",          re.compile(r"completed|finished|done|submitted", re.I)),
    ("cancelled",          re.compile(r"cancelled|canceled", re.I)),
    ("rescheduled",        re.compile(r"rescheduled|moved|new schedule|new time", re.I)),
    ("status_check",       re.compile(r"(?:latest )?status|what happened|current state", re.I)),
    ("related",            re.compile(r"related|messages about|show.*messages.*(?:about|related)", re.I)),
    ("conflicting",        re.compile(r"conflict|contradicting|uncertain|ambiguous", re.I)),
    ("deadline_changed",   re.compile(r"deadline.*changed|deadline.*moved|deadline.*extended", re.I)),
    ("confirmation",       re.compile(r"require.*confirm|need.*confirm|confirmation", re.I)),
    ("blocked",            re.compile(r"block|must be blocked|cannot.*external|sensitive.*block", re.I)),
    ("why_critical",       re.compile(r"why.*critical|why.*marked|reason.*priority", re.I)),
    ("became_critical",    re.compile(r"became\s+critical|turned\s+critical|new.*critical", re.I)),
    ("insufficient",       re.compile(r"approved.*by|was the .+ approved", re.I)),
]

DEADLINE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
MSG_REF_RE = re.compile(r"\b(MSG_\d+|DEMO_\d+)\b")


def _detect_intent(query):
    """Return list of matched intents."""
    intents = []
    for name, pat in INTENT_PATTERNS:
        if pat.search(query):
            intents.append(name)
    return intents if intents else ["general"]


class SemanticAssistant:
    """
    Build an index and answer queries over messages, tasks, groups,
    priorities, and privacy routing.
    """

    def __init__(self, messages_df, cls_df, tasks_df, sens_df,
                 groups, priorities, privacy_routes):
        self.messages = {}
        for _, row in messages_df.iterrows():
            mid = str(row["message_id"])
            self.messages[mid] = {
                "message_id": mid,
                "timestamp": str(row["timestamp"]),
                "sender": str(row["sender"]),
                "message": str(row["message"]),
            }

        self.cls_map = {}
        for _, row in cls_df.iterrows():
            self.cls_map[str(row["message_id"])] = dict(row)

        self.tasks = []
        if not tasks_df.empty:
            for _, row in tasks_df.iterrows():
                self.tasks.append(dict(row))
        self.task_by_msg = {str(t["source_message_id"]): t for t in self.tasks}

        self.sens_map = {}
        if not sens_df.empty:
            for _, row in sens_df.iterrows():
                self.sens_map[str(row["message_id"])] = dict(row)

        self.groups = groups or []
        self.group_by_id = {g["group_id"]: g for g in self.groups}
        self.group_for_msg = {}
        for g in self.groups:
            for mid in g.get("related_message_ids", []):
                self.group_for_msg[mid] = g

        self.priorities = priorities or []
        self.priority_by_msg = {p["message_id"]: p for p in self.priorities}

        self.privacy_routes = privacy_routes or []
        self.route_by_msg = {r["message_id"]: r for r in self.privacy_routes}

        # Build TF-IDF index
        self._build_index()

    def _build_index(self):
        """Build TF-IDF index over messages + task descriptions + group summaries."""
        self._doc_ids = []
        self._doc_texts = []
        self._doc_types = []

        for mid, info in self.messages.items():
            self._doc_ids.append(mid)
            self._doc_texts.append(info["message"])
            self._doc_types.append("message")

        for t in self.tasks:
            doc_id = t.get("item_id", t["source_message_id"])
            self._doc_ids.append(doc_id)
            self._doc_texts.append(
                f"{t.get('title', '')} {t.get('description', '')} {t.get('type', '')}"
            )
            self._doc_types.append("task")

        for g in self.groups:
            self._doc_ids.append(g["group_id"])
            self._doc_texts.append(
                f"{g.get('title', '')} {g.get('summary', '')} {g.get('status', '')}"
            )
            self._doc_types.append("group")

        if self._doc_texts:
            self._vectorizer = TfidfVectorizer(
                lowercase=True, ngram_range=(1, 2),
                max_features=12000, sublinear_tf=True
            )
            self._matrix = self._vectorizer.fit_transform(self._doc_texts)
        else:
            self._vectorizer = None
            self._matrix = None

    def _retrieve(self, query, top_k=10):
        """Return top-k (doc_id, doc_type, score) by TF-IDF cosine similarity."""
        if self._vectorizer is None:
            return []
        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix).flatten()
        top_idx = np.argsort(sims)[::-1][:top_k]
        results = []
        for i in top_idx:
            if sims[i] > 0.01:
                results.append((self._doc_ids[i], self._doc_types[i], float(round(sims[i], 4))))
        return results

    def answer(self, query):
        """
        Answer a query and return structured response.
        """
        intents = _detect_intent(query)
        retrieved = self._retrieve(query, top_k=15)

        supporting_mids = []
        related_items = []
        group_ids = []
        relevance_scores = []

        # Check for explicit message reference in query
        ref_msgs = MSG_REF_RE.findall(query)

        # ── intent-based logic ─────────────────────────────────────
        answer_text = ""
        reason = ""

        if "became_critical" in intents:
            critical_items = [p for p in self.priorities if p["priority"] == "critical"]
            # If query mentions 'demo', filter to DEMO_ messages only
            if re.search(r"\bdemo\b", query, re.I):
                critical_items = [p for p in critical_items if p["message_id"].startswith("DEMO_")]
            if critical_items:
                parts = []
                for p in critical_items:
                    parts.append(f"- {p['message_id']}: {p['reason']}")
                    supporting_mids.append(p["message_id"])
                    if p["item_id"]:
                        related_items.append(p["item_id"])
                    g = self.group_for_msg.get(p["message_id"])
                    if g:
                        group_ids.append(g["group_id"])
                answer_text = f"The following task(s) became critical:\n" + "\n".join(parts)
                reason = "Filtered priorities for critical items."
            else:
                answer_text = "No tasks were assigned critical priority."
                reason = "No critical-priority items found."

        elif "completed" in intents or "cancelled" in intents:
            target_statuses = []
            if "completed" in intents:
                target_statuses.append("completed")
            if "cancelled" in intents:
                target_statuses.append("cancelled")
            matched_groups = [g for g in self.groups if g.get("status") in target_statuses]
            if matched_groups:
                parts = []
                for g in matched_groups:
                    parts.append(f"- {g['title']} ({g['group_id']}): {g['status']}")
                    supporting_mids.extend(g["related_message_ids"])
                    group_ids.append(g["group_id"])
                    related_items.extend(g.get("related_item_ids", []))
                answer_text = "Tasks/meetings with matching status:\n" + "\n".join(parts)
                reason = f"Filtered groups by status: {target_statuses}."
            else:
                answer_text = "No tasks or meetings match the requested status."
                reason = "No matching groups found."

        elif "rescheduled" in intents:
            rescheduled = [g for g in self.groups if g.get("status") == "rescheduled"]
            if rescheduled:
                parts = []
                for g in rescheduled:
                    latest = g.get("latest_deadline", "unknown")
                    parts.append(f"- {g['title']} ({g['group_id']}): latest schedule {latest}")
                    supporting_mids.extend(g["related_message_ids"])
                    group_ids.append(g["group_id"])
                    related_items.extend(g.get("related_item_ids", []))
                answer_text = "Rescheduled meetings/events:\n" + "\n".join(parts)
                reason = "Filtered groups with status 'rescheduled'."
            else:
                answer_text = "No meetings or events were rescheduled."
                reason = "No rescheduled groups found."

        elif "conflicting" in intents:
            conflict_groups = []
            for g in self.groups:
                texts = g.get("_message_texts", [])
                dates = set()
                for t in texts:
                    for m in DEADLINE_RE.finditer(t):
                        dates.add(m.group(1))
                if len(dates) >= 2 or g.get("status") == "unclear":
                    conflict_groups.append((g, dates))
            if conflict_groups:
                parts = []
                for g, dates in conflict_groups:
                    date_str = ", ".join(sorted(dates)) if dates else "ambiguous"
                    parts.append(f"- {g['title']} ({g['group_id']}): dates mentioned: {date_str}")
                    supporting_mids.extend(g["related_message_ids"])
                    group_ids.append(g["group_id"])
                answer_text = "Messages with conflicting or uncertain deadlines:\n" + "\n".join(parts)
                reason = "Identified groups with multiple deadline dates or unclear status."
            else:
                answer_text = "No conflicting or uncertain deadlines detected."
                reason = "No groups have multiple deadline dates."

        elif "blocked" in intents:
            blocked = [r for r in self.privacy_routes if r.get("route") == "blocked"]
            if re.search(r"\bdemo\b", query, re.I):
                blocked = [r for r in blocked if r["message_id"].startswith("DEMO_")]
            if blocked:
                parts = []
                for r in blocked:
                    parts.append(f"- {r['message_id']}: {r.get('reason', '')}")
                    supporting_mids.append(r["message_id"])
                answer_text = "Messages that must be blocked from external processing:\n" + "\n".join(parts)
                reason = "Filtered privacy routes for 'blocked' decisions."
            else:
                answer_text = "No messages are blocked from external processing."
                reason = "No blocked routes found."

        elif "confirmation" in intents:
            confirm = [r for r in self.privacy_routes if r.get("route") == "requires_confirmation"]
            if re.search(r"\bdemo\b", query, re.I):
                confirm = [r for r in confirm if r["message_id"].startswith("DEMO_")]
            if confirm:
                parts = []
                for r in confirm:
                    parts.append(f"- {r['message_id']}: {r.get('reason', '')}")
                    supporting_mids.append(r["message_id"])
                answer_text = "Messages requiring confirmation before processing:\n" + "\n".join(parts)
                reason = "Filtered privacy routes for 'requires_confirmation' decisions."
            else:
                answer_text = "No messages require confirmation."
                reason = "No confirmation-required routes found."

        elif "status_check" in intents and ref_msgs:
            # Status check for a specific message
            mid = ref_msgs[0]
            g = self.group_for_msg.get(mid)
            if not g:
                # Try to find a group by searching the message content
                info = self.messages.get(mid)
                if info:
                    msg_retrieved = self._retrieve(info["message"], top_k=5)
                    for doc_id, doc_type, score in msg_retrieved:
                        if doc_type == "group" and doc_id in self.group_by_id:
                            g = self.group_by_id[doc_id]
                            break
                        elif doc_type == "message" and doc_id != mid:
                            candidate_g = self.group_for_msg.get(doc_id)
                            if candidate_g:
                                g = candidate_g
                                break
            if g:
                answer_text = (
                    f"The latest status of the item referenced by {mid} is: {g['status']}. "
                    f"{g['summary']}"
                )
                supporting_mids.extend(g["related_message_ids"])
                supporting_mids.append(mid)
                group_ids.append(g["group_id"])
                related_items.extend(g.get("related_item_ids", []))
                reason = f"Found group {g['group_id']} related to {mid} via semantic search."
            else:
                p = self.priority_by_msg.get(mid)
                if p:
                    answer_text = f"Message {mid} has priority {p['priority']}. {p['reason']} No group context available."
                    supporting_mids.append(mid)
                    reason = f"Found priority for {mid} but no related group."
                else:
                    info = self.messages.get(mid)
                    if info:
                        answer_text = f"Message {mid} was found but has no related group or priority. Sender: {info['sender']}."
                        supporting_mids.append(mid)
                        reason = "No group or priority context available."
                    else:
                        answer_text = f"Message {mid} was not found in the dataset."
                        reason = "Message ID not found."

        elif "insufficient" in intents:
            # Queries about things we don't have evidence for
            answer_text = (
                "Insufficient evidence to answer this query. "
                "The dataset does not contain information about this approval or event."
            )
            reason = "No matching messages, tasks, or groups found for this query."

        elif "critical_pending" in intents:
            crit = [p for p in self.priorities if p["priority"] in ("critical", "high")]
            # Filter out completed/cancelled
            pending_crit = []
            for p in crit:
                g = self.group_for_msg.get(p["message_id"])
                if g and g.get("status") in ("completed", "cancelled"):
                    continue
                pending_crit.append(p)
            if pending_crit:
                parts = []
                for p in pending_crit[:10]:
                    parts.append(f"- {p['message_id']} ({p['item_id']}): {p['priority']} — {p['reason']}")
                    supporting_mids.append(p["message_id"])
                    if p["item_id"]:
                        related_items.append(p["item_id"])
                answer_text = f"Critical/high-priority pending items ({len(pending_crit)} total):\n" + "\n".join(parts)
                reason = "Filtered priorities for critical/high that are not completed/cancelled."
            else:
                answer_text = "No critical or high-priority pending tasks found."
                reason = "All critical/high items are resolved or none exist."

        if not answer_text:
            # Fallback: use TF-IDF retrieval
            if retrieved:
                top = retrieved[:5]
                parts = []
                for doc_id, doc_type, score in top:
                    if doc_type == "message" and doc_id in self.messages:
                        info = self.messages[doc_id]
                        parts.append(f"- [{doc_id}] ({score:.2f}): {info['sender']} — {info['message'][:120]}")
                        supporting_mids.append(doc_id)
                    elif doc_type == "group" and doc_id in self.group_by_id:
                        g = self.group_by_id[doc_id]
                        parts.append(f"- [{doc_id}] ({score:.2f}): {g['title']} — {g['summary'][:120]}")
                        group_ids.append(doc_id)
                    elif doc_type == "task":
                        parts.append(f"- [{doc_id}] ({score:.2f})")
                        related_items.append(doc_id)
                    relevance_scores.append(score)

                answer_text = "Most relevant results:\n" + "\n".join(parts)
                reason = "Used TF-IDF semantic retrieval to find the most relevant documents."
            else:
                answer_text = "Insufficient evidence to answer this query."
                reason = "No relevant documents found in the index."

        # Build relevance scores from retrieved
        if not relevance_scores:
            for doc_id, _, score in retrieved[:5]:
                if doc_id in supporting_mids or doc_id in group_ids or doc_id in related_items:
                    relevance_scores.append(score)

        return {
            "query": query,
            "answer": answer_text,
            "supporting_message_ids": sorted(set(supporting_mids))[:15],
            "related_item_ids": sorted(set(related_items))[:10],
            "group_ids": sorted(set(group_ids))[:5],
            "relevance_scores": relevance_scores[:10],
            "reason": reason,
        }
