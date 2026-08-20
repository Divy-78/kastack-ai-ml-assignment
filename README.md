# KaStack Message Intelligence — L2

Local, security-first NLP system for the KaStack AI/ML Intern assignment.
**L2 extends L1** with Priority Engine, Related-Message Grouping, Semantic Search, Privacy-Aware Routing, and Benchmarking.

---

## How L2 Extends L1

The L1 system provides:
- Six-category message classification (rule-based + TF-IDF/Logistic Regression)
- Task/event extraction with dates, times, persons, priorities
- Sensitive-data detection and masking

L2 adds **five new modules** on top of L1, processing L1's 900 messages + 180 new L2 messages + 24 demo messages chronologically:

| Feature | Module | Description |
|---------|--------|-------------|
| Priority Engine | `src/priority_engine.py` | Multi-signal priority scoring for actionable messages |
| Message Grouping | `src/grouping_engine.py` | Groups related messages by task/event subject |
| Semantic Search | `src/semantic_search.py` | TF-IDF retrieval + rule-based QA assistant |
| Privacy Router | `src/privacy_router.py` | Three-tier privacy routing |
| Benchmark | `src/benchmark.py` | L1 vs L2 performance comparison |

---

## Architecture

```
Upload CSVs (L1 + L2 + Demo)
  → Chronological merge & sort
  → L1 Pipeline (preserved)
      ├── Sensitive-data shield (regex masking)
      ├── Rule-based classification
      ├── TF-IDF/LR fallback classification
      └── Task/event extraction
  → L2 Extensions
      ├── Related-message grouping (title matching + TF-IDF cosine)
      ├── Multi-signal priority scoring
      ├── Privacy-aware routing
      ├── Semantic search index (TF-IDF)
      └── Benchmark comparison
  → Streamlit UI (10 tabs)
```

---

## How Priority Is Calculated and Updated

Each actionable message (Action Required, Meeting or Event, Sensitive Information) receives a priority score based on **multiple signals**:

| Signal | Weight | Description |
|--------|--------|-------------|
| Deadline proximity | High | Days until deadline (0=today → critical, 1 → high, ≤3 → medium) |
| Overdue | Critical | Past deadline + not completed → always critical |
| Urgent keywords | High | "urgent", "asap", "critical", "high priority" |
| Follow-up pressure | Medium | Multiple follow-up messages → escalate |
| Status change | High | Completed/cancelled → drop to low |
| Deadline change | High | Moved earlier → escalate; extended → de-escalate |
| Sensitivity | Medium | Contains sensitive data → at least medium |
| Response required | Medium | "please confirm", "any update" → bump |
| Sender authority | Low | Project Lead/Mentor/HR → slight bump |
| Ambiguity | Reduce | "might", "may", "not sure" → reduce confidence |

Priority is **re-evaluated** when a later message changes the deadline, urgency, or status via the grouping engine.

The scoring formula uses a blend: `score = 0.6 * max(weights) + 0.4 * mean(weights)`, then maps to `critical ≥ 0.80, high ≥ 0.55, medium ≥ 0.30, low < 0.30`.

Each decision includes: message_id, item_id, priority, reason, signals, confidence.

---

## How Related Messages Are Identified

Messages are grouped using two strategies:

1. **Normalised task/event title matching** (primary, high precision)
   - Extract the subject from each message using regex patterns (e.g., "following up on X", "update: X has been completed", "the deadline to X")
   - Normalise to lowercase alphanumeric
   - Messages with the same normalised subject → same group

2. **TF-IDF cosine similarity** (fallback, for orphan messages)
   - Compute TF-IDF vectors for ungrouped messages
   - Compare against group centroids
   - Assign to group if cosine similarity ≥ 0.35

Within each group:
- Messages are sorted chronologically
- Status is tracked scanning for completion/cancellation/rescheduling/ambiguity patterns
- The latest deadline mentioned across the group is recorded

---

## How Semantic Retrieval Works

The assistant builds a **TF-IDF index** over:
- All messages (masked text)
- Extracted task/event descriptions
- Group summaries

For each query:
1. **Intent detection** — 14 regex patterns identify query intent (tasks today, critical items, rescheduled meetings, status check, blocked messages, etc.)
2. **Structured filtering** — Intent-specific logic filters priorities, groups, privacy routes
3. **TF-IDF retrieval** — Fallback cosine similarity search for general queries
4. **Evidence assembly** — Supporting message IDs, relevance scores, and explanations

If insufficient evidence exists, the assistant explicitly says so rather than generating unsupported answers.

---

## How Privacy-Aware Routing Works

Every message is routed to one of three tiers:

| Tier | Criteria | Action |
|------|----------|--------|
| **Blocked** | High-risk data: passwords, OTPs, PINs, card numbers, bank accounts, recovery codes, tokens, ID numbers | Must not be sent to external services |
| **Requires Confirmation** | Medium-risk: private addresses, contact numbers, health results, personal info category | User must confirm before external use |
| **Process Locally** | No sensitive data detected | Safe for any processing |

The router uses L1's sensitive-data shield results plus additional keyword and category checks.

---

## What Component Was Optimized

The **message grouping engine** was optimized from naive pairwise TF-IDF similarity (O(n²)) to a two-stage approach:
1. **Stage 1** — Deterministic title matching (O(n)) groups most messages instantly
2. **Stage 2** — TF-IDF cosine only for ungrouped messages against group centroids (O(k·m), where k = groups, m = ungrouped)

This reduces the cosine similarity computations from ~1 million (1080² pairwise) to typically ~500 (ungrouped × groups).

---

## How Benchmarking Was Performed

The benchmark compares:
- **Processing time** — Wall-clock `time.perf_counter()` for L1 (900 msgs) vs L2 (combined ~1100 msgs)
- **Data size** — DataFrame memory usage (`memory_usage(deep=True)`)
- **Result counts** — Tasks extracted, groups formed, priorities assigned, sensitive detections
- **Index size** — TF-IDF document collection size

Measured on the same machine in the same session for fair comparison.

---

## Six Required Categories
- Action Required
- Meeting or Event
- Personal Information
- General Information
- Promotional
- Sensitive Information

---

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then upload the CSV files:
1. `messages.csv` (L1 — 900 messages)
2. `mandatory_demo_ids.csv` (L1)
3. `l2_messages.csv` (L2 — 180 messages)
4. `l2_demo_messages.csv` (24 demo messages)
5. `l2_demo_queries.csv` (8 queries)

---

## Output Files

The dashboard provides downloads for:
- `classification_results.csv` — All message classifications
- `mandatory_results.csv` — Mandatory ID results
- `task_event_results.csv` — Extracted tasks and events
- `sensitive_results.csv` — Sensitive data detections
- `priority_results.json` — Priority assignments
- `related_groups.json` — Related message groups
- `privacy_routing.json` — Privacy routing decisions
- `benchmark_report.json` — L1 vs L2 comparison

---

## Assumptions and Limitations

- No answer labels supplied; weak supervision is documented
- Rules are strongest for patterns in this synthetic dataset
- TF-IDF is lightweight but captures keyword semantics, not deep semantics
- Person extraction is conservative (known names only)
- Confidence is not ground-truth calibrated
- Relative dates like "tomorrow" resolved only from message timestamps
- Grouping assumes task titles are consistent across messages
- Priority weights are tuned for this dataset

---

## AI-Tool Usage Declaration

AI tools were used during development for brainstorming, architecture discussion, debugging assistance, code review, and documentation support. The implementation was reviewed and tested locally, and the final system uses custom logic and open-source Python libraries for runtime processing. Raw assignment messages were not sent to external AI services.

---

## Important Data Rule

The supplied L1 and L2 datasets are NOT included in this public repository. The Streamlit app accepts files through uploaders. Do not commit:
- `messages.csv`
- `mandatory_demo_ids.csv`
- `l2_messages.csv`
- `l2_demo_messages.csv`
- `l2_demo_queries.csv`
- Raw screenshots containing sensitive-looking values
- Logs containing unmasked sensitive values
