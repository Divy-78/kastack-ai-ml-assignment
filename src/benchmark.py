"""
Benchmark (L2 Part 5)

Compare L1 vs L2 performance:
  - Processing time
  - Model / index size
  - Result counts and quality
"""

import time
import sys

import pandas as pd


def measure_pipeline_time(pipeline_fn, messages_df, mandatory_ids):
    """Run the pipeline and return (results_tuple, elapsed_seconds)."""
    start = time.perf_counter()
    results = pipeline_fn(messages_df, mandatory_ids)
    elapsed = time.perf_counter() - start
    return results, round(elapsed, 3)


def estimate_object_size(obj):
    """Rough memory estimate in KB."""
    try:
        if isinstance(obj, pd.DataFrame):
            return round(obj.memory_usage(deep=True).sum() / 1024, 1)
        elif isinstance(obj, list):
            import json
            return round(len(json.dumps(obj, default=str).encode()) / 1024, 1)
        else:
            return round(sys.getsizeof(obj) / 1024, 1)
    except Exception:
        return 0.0


def build_benchmark_report(l1_messages_count, l2_messages_count,
                           l1_time, l2_time,
                           l1_tasks_count, l2_tasks_count,
                           l2_groups_count, l2_priorities_count,
                           l1_sensitive_count, l2_sensitive_count,
                           l1_cls_size_kb, l2_cls_size_kb,
                           l2_index_size_kb):
    """
    Build a benchmark comparison dict.
    """
    return {
        "messages": {
            "L1": l1_messages_count,
            "L2 (combined)": l2_messages_count,
            "change": f"+{l2_messages_count - l1_messages_count}",
        },
        "processing_time_sec": {
            "L1": l1_time,
            "L2 (combined)": l2_time,
            "change": f"{l2_time - l1_time:+.3f}s",
        },
        "tasks_events_extracted": {
            "L1": l1_tasks_count,
            "L2 (combined)": l2_tasks_count,
            "change": f"+{l2_tasks_count - l1_tasks_count}",
        },
        "sensitive_detections": {
            "L1": l1_sensitive_count,
            "L2 (combined)": l2_sensitive_count,
            "change": f"+{l2_sensitive_count - l1_sensitive_count}",
        },
        "classification_data_kb": {
            "L1": l1_cls_size_kb,
            "L2 (combined)": l2_cls_size_kb,
        },
        "l2_new_features": {
            "priority_assignments": l2_priorities_count,
            "message_groups": l2_groups_count,
            "search_index_kb": l2_index_size_kb,
        },
    }
