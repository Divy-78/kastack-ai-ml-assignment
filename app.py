
import pandas as pd
import streamlit as st
from src.pipeline import run_pipeline

st.set_page_config(page_title="KaStack Message Intelligence", layout="wide")
st.title("KaStack Message Intelligence")
#st.caption("Local, security-first NLP pipeline for classification, task/event extraction, and sensitive-data protection")

with st.sidebar:
    st.header("Input files")
    st.write("Select the two CSV files to begin.")
    msg_file = st.file_uploader("1. messages.csv", type=["csv"])
    ids_file = st.file_uploader("2. mandatory_demo_ids.csv", type=["csv"])
    st.divider()
    st.caption("Sensitive values are masked before display.")

if not msg_file or not ids_file:
    st.info("Upload both CSV files in the sidebar to start.")
    st.markdown("### Expected columns")
    st.code("messages.csv: message_id, timestamp, sender, message\nmandatory_demo_ids.csv: message_id")
   # st.markdown("### Security rule")
    #st.write("Never commit the supplied dataset or mandatory-ID file to a public GitHub repository.")
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
if messages["message_id"].duplicated().any():
    st.error("Duplicate message_id values detected.")
    st.stop()

try:
    with st.spinner("Processing messages chronologically with the local NLP pipeline..."):
        cls, tasks, sens = run_pipeline(
            messages,
            mandatory["message_id"].dropna().astype(str).tolist()
        )
except Exception as exc:
    st.exception(exc)
    st.stop()

mandatory_ids = mandatory["message_id"].dropna().astype(str).tolist()
mandatory_results = cls[cls["message_id"].isin(mandatory_ids)].copy()
missing_ids = [x for x in mandatory_ids if x not in set(cls["message_id"])]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Messages", len(cls))
c2.metric("Categories", cls["category"].nunique())
c3.metric("Tasks / Events", len(tasks))
c4.metric("Sensitive", len(sens))
c5.metric("Mandatory IDs", f"{len(mandatory_results)}/{len(mandatory_ids)}")

st.success("Pipeline completed locally.")

def csv_bytes(frame):
    return frame.to_csv(index=False).encode("utf-8")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Classification", "Mandatory 15", "Tasks & Events", "Sensitive Shield", "Demo Evidence"
])

with tab1:
    st.subheader("Classification results")
    st.dataframe(
        cls[["message_id", "timestamp", "sender", "category", "confidence", "reason"]],
        width="stretch", height=560
    )
    st.download_button("Download classification_results.csv", csv_bytes(cls),
                       "classification_results.csv", "text/csv")

with tab2:
    st.subheader("Mandatory demonstration IDs")
    if missing_ids:
        st.warning(f"IDs not found in messages.csv: {missing_ids}")
    st.dataframe(
        mandatory_results[["message_id", "category", "confidence", "reason"]],
        width="stretch", height=560
    )
    st.info("Raw message text is intentionally omitted from this table.")
    st.download_button("Download mandatory_15_results.csv", csv_bytes(mandatory_results),
                       "mandatory_15_results.csv", "text/csv")

with tab3:
    st.subheader("Extracted tasks and events")
    if tasks.empty:
        st.warning("No task/event items were extracted.")
    else:
        st.dataframe(tasks, width="stretch", height=560)
        st.download_button("Download task_event_results.csv", csv_bytes(tasks),
                           "task_event_results.csv", "text/csv")

with tab4:
    st.subheader("Sensitive information shield")
    if sens.empty:
        st.success("No sensitive-information patterns were detected.")
    else:
        st.dataframe(sens, width="stretch", height=560)
        st.download_button("Download sensitive_results.csv", csv_bytes(sens),
                           "sensitive_results.csv", "text/csv")
    st.warning("Sensitive values are masked before display. Do not send raw messages to external AI services.")

with tab5:
    st.subheader("Demo evidence")
    counts = cls["category"].value_counts().reindex([
        "Action Required", "Meeting or Event", "Personal Information",
        "General Information", "Promotional", "Sensitive Information"
    ], fill_value=0)
    st.dataframe(counts.rename("messages").reset_index(), width="stretch")
    if not tasks.empty:
        st.markdown("**Extraction coverage**")
        st.dataframe(tasks.groupby("type").size().rename("count").reset_index(), width="stretch")
    st.markdown("**Methodology note**")
    st.write(
        "The supplied dataset has no ground-truth answer labels. The project therefore "
        "uses transparent weak supervision. Confidence is a rule/model confidence signal, "
        "not a validated accuracy score."
    )

