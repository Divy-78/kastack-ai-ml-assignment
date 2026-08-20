"""
KaStack Message Intelligence — L2 Extended Application

Extends the L1 Streamlit app with:
  - Priority & Action Engine
  - Related-Message Grouping
  - Semantic Search & Intelligent Assistant
  - Privacy-Aware Routing
  - Benchmark Comparison
"""

import json
import time

import pandas as pd
import streamlit as st

from src.pipeline import run_pipeline
from src.priority_engine import assign_priorities
from src.grouping_engine import build_groups
from src.semantic_search import SemanticAssistant
from src.privacy_router import route_all_messages
from src.benchmark import measure_pipeline_time, estimate_object_size, build_benchmark_report

# ── page config ─────────────────────────────────────────────────────

st.set_page_config(page_title="KaStack Message Intelligence — L2", layout="wide")
st.title("KaStack Message Intelligence")
st.caption("L2 — Priority Engine · Message Grouping · Semantic Assistant · Privacy Router")

# ── sidebar ─────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Input Files")

    st.subheader("L1 Dataset")
    msg_file = st.file_uploader("messages.csv (L1 — 900 messages)", type=["csv"], key="l1_msg")
    ids_file = st.file_uploader("mandatory_demo_ids.csv (L1)", type=["csv"], key="l1_ids")

    st.subheader("L2 Dataset")
    l2_msg_file = st.file_uploader("l2_messages.csv (180 messages)", type=["csv"], key="l2_msg")

    st.subheader("L2 Demo Files")
    demo_msg_file = st.file_uploader("l2_demo_messages.csv (24 demo)", type=["csv"], key="demo_msg")
    demo_query_file = st.file_uploader("l2_demo_queries.csv", type=["csv"], key="demo_q")

    st.divider()
    st.caption("Sensitive values are masked before display.")

# ── validate inputs ─────────────────────────────────────────────────

if not msg_file or not ids_file:
    st.info("Upload the L1 CSV files in the sidebar to start. L2 files are optional for extended features.")
    st.markdown("### Expected columns")
    st.code("messages.csv: message_id, timestamp, sender, message\nmandatory_demo_ids.csv: message_id")
    st.stop()

try:
    messages = pd.read_csv(msg_file)
    mandatory = pd.read_csv(ids_file)
except Exception as exc:
    st.error(f"Could not read an uploaded CSV: {exc}")
    st.stop()

required = {"message_id", "timestamp", "sender", "message"}
missing = required - set(messages.columns)
if missing:
    st.error(f"messages.csv is missing required columns: {sorted(missing)}")
    st.stop()
if "message_id" not in mandatory.columns:
    st.error("mandatory_demo_ids.csv must contain a message_id column.")
    st.stop()

# ── merge L2 messages ──────────────────────────────────────────────

l2_messages = None
if l2_msg_file:
    try:
        l2_messages = pd.read_csv(l2_msg_file)
    except Exception as exc:
        st.warning(f"Could not read L2 messages: {exc}")

demo_messages = None
if demo_msg_file:
    try:
        demo_messages = pd.read_csv(demo_msg_file)
    except Exception as exc:
        st.warning(f"Could not read demo messages: {exc}")

demo_queries = None
if demo_query_file:
    try:
        demo_queries = pd.read_csv(demo_query_file)
    except Exception as exc:
        st.warning(f"Could not read demo queries: {exc}")

# Build combined dataset (L1 + L2 + demo, chronological)
l1_count = len(messages)
all_messages = messages.copy()
if l2_messages is not None:
    all_messages = pd.concat([all_messages, l2_messages], ignore_index=True)
if demo_messages is not None:
    all_messages = pd.concat([all_messages, demo_messages], ignore_index=True)

# De-duplicate by message_id (keep first)
all_messages = all_messages.drop_duplicates(subset="message_id", keep="first")
all_messages["timestamp"] = pd.to_datetime(all_messages["timestamp"], errors="coerce")
all_messages = all_messages.sort_values("timestamp", kind="stable").reset_index(drop=True)

combined_count = len(all_messages)

# ── run pipelines ──────────────────────────────────────────────────

mandatory_ids = mandatory["message_id"].dropna().astype(str).tolist()

try:
    with st.spinner("Running L1 pipeline on original messages (for benchmark)..."):
        l1_start = time.perf_counter()
        l1_cls, l1_tasks, l1_sens = run_pipeline(messages, mandatory_ids)
        l1_time = round(time.perf_counter() - l1_start, 3)

    with st.spinner(f"Running full pipeline on {combined_count} messages (L1 + L2)..."):
        l2_start = time.perf_counter()
        cls, tasks, sens = run_pipeline(all_messages, mandatory_ids)
        l2_time = round(time.perf_counter() - l2_start, 3)

except Exception as exc:
    st.exception(exc)
    st.stop()

# Attach message text to cls for downstream use
cls = cls.merge(
    all_messages[["message_id", "message"]],
    on="message_id", how="left", suffixes=("", "_raw")
)

# ── L2 engines ─────────────────────────────────────────────────────

with st.spinner("Building related-message groups..."):
    groups = build_groups(cls, tasks, all_messages)

with st.spinner("Assigning priorities..."):
    priorities = assign_priorities(cls, tasks, sens, groups)

with st.spinner("Routing privacy decisions..."):
    privacy_routes = route_all_messages(cls, sens, all_messages)

with st.spinner("Building semantic search index..."):
    assistant = SemanticAssistant(
        all_messages, cls, tasks, sens, groups, priorities, privacy_routes
    )

# ── metrics ─────────────────────────────────────────────────────────

st.success(f"Pipeline completed — {combined_count} messages processed locally.")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Messages", combined_count)
c2.metric("Categories", cls["category"].nunique())
c3.metric("Tasks / Events", len(tasks))
c4.metric("Sensitive", len(sens))
c5.metric("Groups", len(groups))
c6.metric("Priorities", len(priorities))

# ── helpers ─────────────────────────────────────────────────────────

def csv_bytes(frame):
    return frame.to_csv(index=False).encode("utf-8")

def json_bytes(data):
    return json.dumps(data, indent=2, default=str).encode("utf-8")

# ── tabs ────────────────────────────────────────────────────────────

tab_names = [
    "Classification", "Mandatory IDs", "Tasks & Events", "Sensitive Shield",
    "Priority Engine", "Message Groups", "AI Assistant",
    "Privacy Router", "Benchmark", "Demo Evidence"
]
tabs = st.tabs(tab_names)

# ── Tab 1: Classification ──────────────────────────────────────────
with tabs[0]:
    st.subheader("Classification Results")
    display_cols = ["message_id", "timestamp", "sender", "category", "confidence", "reason"]
    st.dataframe(cls[display_cols], use_container_width=True, height=560)
    st.download_button("Download classification_results.csv", csv_bytes(cls[display_cols]),
                       "classification_results.csv", "text/csv")

# ── Tab 2: Mandatory IDs ──────────────────────────────────────────
with tabs[1]:
    st.subheader("Mandatory Demonstration IDs")
    mandatory_results = cls[cls["message_id"].isin(mandatory_ids)].copy()
    missing_ids = [x for x in mandatory_ids if x not in set(cls["message_id"])]
    if missing_ids:
        st.warning(f"IDs not found: {missing_ids}")
    st.dataframe(mandatory_results[["message_id", "category", "confidence", "reason"]],
                 use_container_width=True, height=560)
    st.info("Raw message text is intentionally omitted from this table.")
    st.download_button("Download mandatory_results.csv", csv_bytes(mandatory_results),
                       "mandatory_results.csv", "text/csv")

# ── Tab 3: Tasks & Events ─────────────────────────────────────────
with tabs[2]:
    st.subheader("Extracted Tasks and Events")
    if tasks.empty:
        st.warning("No task/event items were extracted.")
    else:
        st.dataframe(tasks, use_container_width=True, height=560)
        st.download_button("Download task_event_results.csv", csv_bytes(tasks),
                           "task_event_results.csv", "text/csv")

# ── Tab 4: Sensitive Shield ────────────────────────────────────────
with tabs[3]:
    st.subheader("Sensitive Information Shield")
    if sens.empty:
        st.success("No sensitive-information patterns were detected.")
    else:
        st.dataframe(sens, use_container_width=True, height=560)
        st.download_button("Download sensitive_results.csv", csv_bytes(sens),
                           "sensitive_results.csv", "text/csv")
    st.warning("Sensitive values are masked before display. Do not send raw messages to external AI services.")

# ── Tab 5: Priority Engine ────────────────────────────────────────
with tabs[4]:
    st.subheader("Priority & Action Engine")
    st.markdown("Every actionable message receives a multi-signal priority score.")

    if priorities:
        prio_df = pd.DataFrame(priorities)
        # Convert signals list to string for display
        prio_df["signals"] = prio_df["signals"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))

        # Filters
        col_a, col_b = st.columns(2)
        with col_a:
            prio_filter = st.multiselect("Filter by priority", ["critical", "high", "medium", "low"],
                                         default=["critical", "high", "medium", "low"])
        with col_b:
            search_prio = st.text_input("Search message ID", key="prio_search")

        filtered = prio_df[prio_df["priority"].isin(prio_filter)]
        if search_prio:
            filtered = filtered[filtered["message_id"].str.contains(search_prio, case=False)]

        st.dataframe(filtered, use_container_width=True, height=560)

        # Stats
        st.markdown("#### Priority Distribution")
        stats = prio_df["priority"].value_counts().reindex(
            ["critical", "high", "medium", "low"], fill_value=0
        )
        st.bar_chart(stats)

        st.download_button("Download priority_results.json", json_bytes(priorities),
                           "priority_results.json", "application/json")
    else:
        st.info("No actionable messages to prioritize.")

# ── Tab 6: Message Groups ─────────────────────────────────────────
with tabs[5]:
    st.subheader("Related-Message Groups")
    st.markdown("Messages referring to the same task, meeting, or subject are grouped together.")

    if groups:
        # Summary table
        group_summary = []
        for g in groups:
            group_summary.append({
                "group_id": g["group_id"],
                "title": g["title"],
                "messages": len(g["related_message_ids"]),
                "status": g["status"],
                "latest_deadline": g.get("latest_deadline", "—"),
                "confidence": g["confidence"],
            })
        gs_df = pd.DataFrame(group_summary)
        st.dataframe(gs_df, use_container_width=True, height=300)

        # Detail expanders
        for g in groups:
            with st.expander(f"{g['group_id']}: {g['title']} ({g['status']})"):
                st.markdown(f"**Summary:** {g['summary']}")
                st.markdown(f"**Status:** `{g['status']}`  |  **Deadline:** `{g.get('latest_deadline', '—')}`  |  **Confidence:** `{g['confidence']}`")
                st.markdown(f"**Message IDs:** {', '.join(g['related_message_ids'][:20])}")
                if g.get("related_item_ids"):
                    st.markdown(f"**Related Items:** {', '.join(g['related_item_ids'])}")

        # Clean groups for JSON export (remove internal _message_texts)
        export_groups = []
        for g in groups:
            eg = {k: v for k, v in g.items() if not k.startswith("_")}
            export_groups.append(eg)

        st.download_button("Download related_groups.json", json_bytes(export_groups),
                           "related_groups.json", "application/json")
    else:
        st.info("No message groups identified.")

# ── Tab 7: AI Assistant ───────────────────────────────────────────
with tabs[6]:
    st.subheader("Semantic Search & Intelligent Assistant")

    # Demo queries
    if demo_queries is not None and not demo_queries.empty:
        st.markdown("#### Mandatory Demo Queries")
        for _, qrow in demo_queries.iterrows():
            qid = qrow.get("query_id", "")
            query_text = qrow.get("query", "")
            with st.expander(f"{qid}: {query_text}"):
                result = assistant.answer(query_text)
                st.markdown(f"**Answer:** {result['answer']}")
                st.markdown(f"**Supporting Messages:** {', '.join(result['supporting_message_ids'][:10])}")
                if result.get("related_item_ids"):
                    st.markdown(f"**Related Items:** {', '.join(result['related_item_ids'][:5])}")
                if result.get("group_ids"):
                    st.markdown(f"**Groups:** {', '.join(result['group_ids'][:3])}")
                if result.get("relevance_scores"):
                    st.markdown(f"**Relevance Scores:** {result['relevance_scores'][:5]}")
                st.markdown(f"**Reason:** {result['reason']}")
                st.json(result)

    st.markdown("---")
    st.markdown("#### Ask a Question")
    user_query = st.text_input("Enter your query:", placeholder="e.g., What tasks should I complete today?")
    if user_query:
        with st.spinner("Searching..."):
            result = assistant.answer(user_query)
        st.markdown(f"**Answer:** {result['answer']}")
        st.markdown(f"**Supporting Messages:** {', '.join(result['supporting_message_ids'][:10])}")
        if result.get("related_item_ids"):
            st.markdown(f"**Related Items:** {', '.join(result['related_item_ids'][:5])}")
        if result.get("group_ids"):
            st.markdown(f"**Groups:** {', '.join(result['group_ids'][:3])}")
        if result.get("relevance_scores"):
            st.markdown(f"**Relevance Scores:** {result['relevance_scores'][:5]}")
        st.markdown(f"**Reason:** {result['reason']}")
        with st.expander("Full JSON response"):
            st.json(result)

# ── Tab 8: Privacy Router ─────────────────────────────────────────
with tabs[7]:
    st.subheader("Privacy-Aware Routing")
    st.markdown("Each message is routed to one of three tiers: **process locally**, **requires confirmation**, or **blocked**.")

    if privacy_routes:
        pr_df = pd.DataFrame(privacy_routes)

        # Stats
        route_counts = pr_df["route"].value_counts()
        col_x, col_y, col_z = st.columns(3)
        col_x.metric("Process Locally", route_counts.get("process_locally", 0))
        col_y.metric("Requires Confirmation", route_counts.get("requires_confirmation", 0))
        col_z.metric("Blocked", route_counts.get("blocked", 0))

        # Filter
        route_filter = st.multiselect(
            "Filter by route",
            ["process_locally", "requires_confirmation", "blocked"],
            default=["requires_confirmation", "blocked"]
        )
        filtered_pr = pr_df[pr_df["route"].isin(route_filter)]
        st.dataframe(filtered_pr, use_container_width=True, height=560)

        st.download_button("Download privacy_routing.json", json_bytes(privacy_routes),
                           "privacy_routing.json", "application/json")
    else:
        st.info("No privacy routing results.")

# ── Tab 9: Benchmark ──────────────────────────────────────────────
with tabs[8]:
    st.subheader("Benchmark Comparison: L1 vs L2")

    benchmark = build_benchmark_report(
        l1_messages_count=l1_count,
        l2_messages_count=combined_count,
        l1_time=l1_time,
        l2_time=l2_time,
        l1_tasks_count=len(l1_tasks),
        l2_tasks_count=len(tasks),
        l2_groups_count=len(groups),
        l2_priorities_count=len(priorities),
        l1_sensitive_count=len(l1_sens),
        l2_sensitive_count=len(sens),
        l1_cls_size_kb=estimate_object_size(l1_cls),
        l2_cls_size_kb=estimate_object_size(cls),
        l2_index_size_kb=estimate_object_size(assistant._doc_texts),
    )

    st.markdown("#### Processing Metrics")
    bm = benchmark
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Messages**")
        st.json(bm["messages"])
        st.markdown("**Processing Time**")
        st.json(bm["processing_time_sec"])
    with col2:
        st.markdown("**Tasks/Events Extracted**")
        st.json(bm["tasks_events_extracted"])
        st.markdown("**Sensitive Detections**")
        st.json(bm["sensitive_detections"])

    st.markdown("#### L2 New Features")
    st.json(bm["l2_new_features"])

    st.markdown("#### Data Sizes")
    st.json(bm["classification_data_kb"])

    st.download_button("Download benchmark_report.json", json_bytes(benchmark),
                       "benchmark_report.json", "application/json")

# ── Tab 10: Demo Evidence ─────────────────────────────────────────
with tabs[9]:
    st.subheader("Demo Evidence")

    # Category distribution
    counts = cls["category"].value_counts().reindex([
        "Action Required", "Meeting or Event", "Personal Information",
        "General Information", "Promotional", "Sensitive Information"
    ], fill_value=0)
    st.dataframe(counts.rename("messages").reset_index(), use_container_width=True)

    if not tasks.empty:
        st.markdown("**Extraction coverage**")
        st.dataframe(tasks.groupby("type").size().rename("count").reset_index(), use_container_width=True)

    # L2 evidence
    st.markdown("---")
    st.markdown("**L2 Extension Summary**")
    st.markdown(f"""
    - **Priority assignments:** {len(priorities)} messages scored
    - **Message groups:** {len(groups)} groups identified
    - **Privacy routes:** {len(privacy_routes)} messages routed ({len([r for r in privacy_routes if r['route'] == 'blocked'])} blocked, {len([r for r in privacy_routes if r['route'] == 'requires_confirmation'])} require confirmation)
    - **Search index:** {len(assistant._doc_ids)} documents indexed
    - **L1 processing time:** {l1_time:.3f}s | **L2 processing time:** {l2_time:.3f}s
    """)

    st.markdown("**Methodology note**")
    st.write(
        "The supplied dataset has no ground-truth answer labels. The project uses "
        "transparent weak supervision. Confidence is a rule/model confidence signal, "
        "not a validated accuracy score. All processing is local."
    )
