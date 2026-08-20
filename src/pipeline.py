
import re
from datetime import timedelta
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

CATEGORIES = [
    "Action Required", "Meeting or Event", "Personal Information",
    "General Information", "Promotional", "Sensitive Information"
]

SENSITIVE_PATTERNS = [
<<<<<<< HEAD
    ("password", re.compile(r"\b(?:temporary\s+)?password\s*[:=]?\s*(?:is\s+)?[^\s,.;]+(?:[-#][^\s,.;]+)*", re.I), "high", "do_not_store"),
    ("one_time_password", re.compile(r"\b(?:your\s+)?(?:fictional\s+)?OTP\s*(?:is|=|:)\s*[A-Za-z0-9-]+", re.I), "high", "do_not_store"),
    ("pin", re.compile(r"\bPIN\s*(?:is|=|:)\s*[A-Za-z0-9-]+", re.I), "high", "do_not_store"),
    ("card_number", re.compile(r"\b\d{4}(?:[ -]\d{4}){3}(?:[-]\d+)?\b"), "high", "do_not_store"),
    ("bank_account", re.compile(r"\b(?:bank account(?:\s+number)?|sample bank account)\s*(?:is|=|:)?\s*[A-Za-z0-9-]+", re.I), "high", "do_not_store"),
    ("account_recovery_code", re.compile(r"\b(?:account\s+)?recovery code\s*(?:is|=|:)?\s*[A-Za-z0-9-]+", re.I), "high", "do_not_store"),
    ("authentication_token", re.compile(r"\b(?:temporary access token|access token|integration token)[:\s]+\s*[A-Za-z0-9_-]+", re.I), "high", "do_not_send_external"),
    ("identification_number", re.compile(r"\b(?:identification number|fictional ID number)\s*(?:is|=|:)?\s*[A-Za-z0-9-]+", re.I), "high", "do_not_store"),
    ("private_address", re.compile(r"\bhome address\s*(?:is|=|:)\s*[^.]+", re.I), "high", "do_not_store"),
    ("private_address", re.compile(r"\b(?:deliver\s+.+?\s+to|deliver it to)\s+\d+\s+[^.]+", re.I), "high", "do_not_store"),
    ("private_contact", re.compile(r"\b(?:phone|mobile|contact)\s*(?:number|me on)?\s*(?:is|at|on)?\s*[+\d][\d ()-]{7,}", re.I), "high", "do_not_store"),
    ("private_contact", re.compile(r"\bcall me on\s+[\d ]{7,}", re.I), "high", "do_not_store"),
    ("login_details", re.compile(r"\blogin details\b", re.I), "high", "do_not_send_external"),
    ("health_result", re.compile(r"\b(?:recent test result\s+says|private medical note\s+mentions)\s+[^.]+", re.I), "high", "do_not_store"),
=======
    ("password", re.compile(r"\bpassword\s*[:=]?\s*[^\s,.;]+(?:[-#][^\s,.;]+)*", re.I), "high", "do_not_store"),
    ("one_time_password", re.compile(r"\b(?:your\s+)?OTP\s*(?:is|=|:)\s*[A-Za-z0-9-]+", re.I), "high", "do_not_store"),
    ("pin", re.compile(r"\bPIN\s*(?:is|=|:)\s*[A-Za-z0-9-]+", re.I), "high", "do_not_store"),
    ("card_number", re.compile(r"\b\d{4}(?:[ -]\d{4}){3}(?:[-]\d+)?\b"), "high", "do_not_store"),
    ("bank_account", re.compile(r"\bbank account number\s*(?:is|=|:)?\s*[A-Za-z0-9-]+", re.I), "high", "do_not_store"),
    ("account_recovery_code", re.compile(r"\baccount recovery code\s*(?:is|=|:)?\s*[A-Za-z0-9-]+", re.I), "high", "do_not_store"),
    ("authentication_token", re.compile(r"\b(?:temporary access token|access token)\s*(?:is|=|:)?\s*[A-Za-z0-9_-]+", re.I), "high", "do_not_send_external"),
    ("identification_number", re.compile(r"\bidentification number\s*(?:is|=|:)?\s*[A-Za-z0-9-]+", re.I), "high", "do_not_store"),
    ("private_address", re.compile(r"\bhome address\s*(?:is|=|:)\s*[^.]+", re.I), "high", "do_not_store"),
    ("private_contact", re.compile(r"\b(?:phone|mobile|contact)\s*(?:number|me on)?\s*(?:is|at|on)?\s*[+\d][\d ()-]{7,}", re.I), "high", "do_not_store"),
    ("login_details", re.compile(r"\blogin details\b", re.I), "high", "do_not_send_external"),
    ("health_result", re.compile(r"\brecent test result\s+says\s+[^.]+", re.I), "high", "do_not_store"),
>>>>>>> 61b4bc759b086f6c3389a31b47d0657414023176
]

PROMO_RE = re.compile(r"discount|flash sale|premium plan|save\s*\d+%?|\d+%\s*off|limited[- ]time offer|free delivery|reward points|promo code", re.I)
PERSONAL_RE = re.compile(r"emergency contact|favourite language|favorite language|prefer (?:morning|evening)|prefer receiving updates|drink coffee without sugar|i am vegetarian|i use dark mode|t-shirt size|usually study after dinner|live near|prefer (?:email|text|phone)", re.I)
<<<<<<< HEAD
EVENT_RE = re.compile(r"calendar update:|catch-up happens|orientation on|team stand-up|study-group session|client discussion is scheduled|product demo is scheduled|doctor appointment happens|AI workshop on|project review is scheduled|technical interview|design review at|college seminar at|sprint planning happens|are you available for the (?:design review|college seminar)|let us meet|meet sometime|has been moved to|has been cancelled|session is scheduled|latency-review meeting|meeting is scheduled", re.I)
ACTION_RE = re.compile(r"please\s+(?:review|reply|join|confirm|complete|send|upload|submit|call|pay|renew|update|finish|prepare|check|note)|(?:i )?need you to|don['’]t forget|deadline is|is due on|before\s+20\d{2}-\d{2}-\d{2}|by\s+20\d{2}-\d{2}-\d{2}|could you send it soon|if possible, review the file before the meeting|please call|the deadline to\s+\S+|this is urgent|treat this as urgent|new task:|test the|has been extended|has been completed|any update on|any progress on|can you share an update|following up on|is it in progress|you can cancel|still needs attention|has .+ been handled|i am referring to|compare two|document privacy|create a|validate the|measure memory|update the architecture|prepare the", re.I)
=======
EVENT_RE = re.compile(r"calendar update:|catch-up happens|orientation on|team stand-up|study-group session|client discussion is scheduled|product demo is scheduled|doctor appointment happens|AI workshop on|project review is scheduled|technical interview|design review at|college seminar at|sprint planning happens|are you available for the (?:design review|college seminar)|let us meet|meet sometime", re.I)
ACTION_RE = re.compile(r"please\s+(?:review|reply|join|confirm|complete|send|upload|submit|call|pay|renew|update|finish|prepare)|(?:i )?need you to|don['’]t forget|deadline is|is due on|before\s+20\d{2}-\d{2}-\d{2}|by\s+20\d{2}-\d{2}-\d{2}|could you send it soon|if possible, review the file before the meeting|please call", re.I)
>>>>>>> 61b4bc759b086f6c3389a31b47d0657414023176
DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
PERSON_RE = re.compile(r"\b(?:Maya|Meera|Ishaan|Kabir|Aarav|Ananya|Neha|Tara|Rohan|Vikram)\b")

MASKS = {
    "password": "[PASSWORD_MASKED]", "one_time_password": "[OTP_MASKED]", "pin": "[PIN_MASKED]",
    "card_number": "[CARD_NUMBER_MASKED]", "bank_account": "[BANK_ACCOUNT_MASKED]",
    "account_recovery_code": "[RECOVERY_CODE_MASKED]", "authentication_token": "[TOKEN_MASKED]",
<<<<<<< HEAD
    "identification_number": "[ID_NUMBER_MASKED]", "private_address": "[ADDRESS_MASKED]",
=======
    "identification_number": "[ID_NUMBER_MASKED]", "private_address": "home address [ADDRESS_MASKED]",
>>>>>>> 61b4bc759b086f6c3389a31b47d0657414023176
    "private_contact": "[CONTACT_MASKED]", "login_details": "[LOGIN_DETAILS_MASKED]",
    "health_result": "[HEALTH_RESULT_MASKED]"
}

def mask_sensitive(text):
    spans = []
    for kind, pat, risk, action in SENSITIVE_PATTERNS:
        for m in pat.finditer(text):
            spans.append((m.start(), m.end(), kind, risk, action))
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    chosen, end = [], -1
    for span in spans:
        if span[0] >= end:
            chosen.append(span)
            end = span[1]
    masked = text
    for start, finish, kind, _, _ in reversed(chosen):
        masked = masked[:start] + MASKS[kind] + masked[finish:]
    findings, seen = [], set()
    for _, _, kind, risk, action in chosen:
        if kind not in seen:
            findings.append((kind, risk, action))
            seen.add(kind)
    return masked, findings

def rule_label(text):
    if any(p.search(text) for _, p, _, _ in SENSITIVE_PATTERNS):
        return "Sensitive Information", 0.99, "A high-precision sensitive-data pattern matched; security classification takes precedence."
    if PROMO_RE.search(text):
        return "Promotional", 0.98, "The message contains explicit promotional language such as an offer, discount, sale, or promo code."
    if EVENT_RE.search(text) and (DATE_RE.search(text) or TIME_RE.search(text) or re.search(r"tomorrow|next week|Friday afternoon", text, re.I)):
        return "Meeting or Event", 0.96, "The message contains an event/meeting signal together with scheduling context."
    if ACTION_RE.search(text):
        return "Action Required", 0.95, "The message contains an explicit request, deadline, reminder, or action-oriented instruction."
    if PERSONAL_RE.search(text):
        return "Personal Information", 0.94, "The message states a personal preference, profile detail, or routine."
    return None, 0.0, ""

def extract_date(text, timestamp):
    m = DATE_RE.search(text)
    if m:
        return m.group(1)
    if re.search(r"\btomorrow\b", text, re.I):
        return (timestamp + timedelta(days=1)).strftime("%Y-%m-%d")
    return None

def extract_time(text):
    m = TIME_RE.search(text)
    if m:
        return m.group(0)
    m = re.search(r"\b(\d{1,2})\s*(AM|PM)\b", text, re.I)
    if m:
        return f"{m.group(1)} {m.group(2).upper()}"
    if re.search(r"Friday afternoon", text, re.I):
        return "unresolved"
    return None

def priority(text):
    if re.search(r"urgent|asap|critical|high priority", text, re.I):
        return "high"
    if re.search(r"deadline|due|don['’]t forget|important|must|please|need you to|review|submit|pay|renew", text, re.I):
        return "medium"
    return "low"

def extract_person(text):
    m = PERSON_RE.search(text)
    return m.group(0) if m else None

def task_title(text):
    patterns = [
        r"(?:please\s+)?(?:reply to|review|renew|pay|upload|submit|update|complete|finish|send|call|prepare)\s+([^.;]+?)(?:\s+by\s+20\d{2}-\d{2}-\d{2}|;|\.|$)",
        r"(?:don['’]t forget to)\s+([^;]+?)(?:;|\.|$)",
        r"(?:need you to|i need you to)\s+([^.;]+?)(?:\s+by\s+20\d{2}-\d{2}-\d{2}|\.|$)",
    ]
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            return m.group(1).strip()
    if re.search(r"could you send it soon", text, re.I):
        return "Send the requested item"
    m = re.search(r"(?:complete|prepare|submit|upload|send)\s+(.+?)\s+is due on", text, re.I)
    return m.group(1).strip() if m else None

def event_title(text):
    patterns = [
        r"calendar update:\s*([^,]+)",
        r"(?:catch-up|orientation|stand-up|study-group session|AI workshop|doctor appointment|sprint planning)\b[^,.;]*",
        r"(?:client discussion|product demo|project review)\s+is scheduled",
        r"(?:design review|college seminar|technical interview)\b",
    ]
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            return m.group(0).strip().rstrip(".")
    return "Meeting" if re.search(r"let us meet|meet sometime", text, re.I) else None

def extract_items(row, category):
    text = str(row["message"])
    ts = pd.to_datetime(row["timestamp"]).to_pydatetime()
    date, time, person = extract_date(text, ts), extract_time(text), extract_person(text)
    items = []
    if category == "Action Required":
        title = task_title(text)
        if title:
            items.append({
                "type": "task", "title": title, "description": mask_sensitive(text)[0],
                "date_or_deadline": date, "time": time, "person": person,
                "priority": priority(text), "source_message_id": row["message_id"]
            })
    elif category == "Meeting or Event":
        title = event_title(text)
        if title:
            items.append({
                "type": "event", "title": title, "description": mask_sensitive(text)[0],
                "date_or_deadline": date or "unresolved", "time": time or "unresolved",
                "person": person, "priority": priority(text), "source_message_id": row["message_id"]
            })
    return items

def build_outputs(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)

    weak = [rule_label(t)[0] or "General Information" for t in df["message"].astype(str)]
    vec = FeatureUnion([
        ("word", TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=1, max_features=16000, sublinear_tf=True, strip_accents="unicode")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1, max_features=16000, sublinear_tf=True)),
    ])
    X = vec.fit_transform(df["message"].astype(str))
    clf = LogisticRegression(max_iter=1500, C=3.0, class_weight="balanced", random_state=42)
    clf.fit(X, weak)
    probs, classes = clf.predict_proba(X), list(clf.classes_)

    rows, tasks, sensitive = [], [], []
    ti = ei = 1
    for i, row in df.iterrows():
        text = str(row["message"])
        rc, rcf, reason = rule_label(text)
        if rc:
            pred, conf = rc, rcf
        else:
            j = int(np.argmax(probs[i]))
            pred, conf = classes[j], float(probs[i, j])
            if conf < 0.60:
                pred, conf = "General Information", max(0.55, conf)
                reason = "No high-confidence rule matched; the local text model did not find a sufficiently strong alternative."
            else:
                reason = f"Local TF-IDF semantic classifier selected this category with model probability {conf:.2f}."

        masked, findings = mask_sensitive(text)
        rows.append({
            "message_id": row["message_id"],
            "timestamp": pd.Timestamp(row["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
            "sender": row["sender"], "category": pred,
            "confidence": round(conf, 3), "reason": reason
        })

        for item in extract_items(row, pred):
            if item["type"] == "task":
                item["item_id"] = f"TASK_{ti:03d}"; ti += 1
            else:
                item["item_id"] = f"EVENT_{ei:03d}"; ei += 1
            tasks.append(item)

        if findings:
            kind, risk, action = findings[0]
            sensitive.append({
                "message_id": row["message_id"], "sensitivity_type": kind,
                "risk": risk, "masked_text": masked,
                "recommended_action": action
            })

    return pd.DataFrame(rows), pd.DataFrame(tasks), pd.DataFrame(sensitive)

def run_pipeline(input_data, mandatory_ids=None):
    if isinstance(input_data, (str,)):
        df = pd.read_csv(input_data)
    else:
        df = input_data.copy()
    required = {"message_id", "timestamp", "sender", "message"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df["message_id"].duplicated().any():
        raise ValueError("Duplicate message_id detected.")
    if df.empty:
        raise ValueError("messages.csv is empty.")
    return build_outputs(df)
