import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [ADVANCED-AI] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="InsightFlow advanced analysis runner")
    parser.add_argument("--dataset", required=True, help="Path to a JSON dataset")
    parser.add_argument("--output_dir", default=None, help="Directory for generated reports")
    return parser.parse_args()


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def unwrap_items(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if not isinstance(raw, dict):
        return []

    for key in ("items", "documents", "data", "mentions"):
        value = raw.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]

    if isinstance(raw.get("articles"), list) or isinstance(raw.get("tweets"), list):
        items = []
        items.extend(x for x in raw.get("articles", []) if isinstance(x, dict))
        items.extend(x for x in raw.get("tweets", []) if isinstance(x, dict))
        return items

    return []


def normalize_source(source: str) -> str:
    s = (source or "").strip().lower()
    if s in {"news", "press", "presse", "article", "actualites", "actualités"}:
        return "press"
    if s in {"twitter", "twitter/x", "x", "tweet"}:
        return "twitter"
    if s == "reddit":
        return "twitter"
    return s or "unknown"


def normalize_date(value: Any) -> str:
    if not value:
        return datetime.now().isoformat()
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).isoformat()
    except ValueError:
        return str(value)


def normalize_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for idx, item in enumerate(items):
        content = (
            item.get("content")
            or item.get("text")
            or item.get("body")
            or item.get("description")
            or item.get("title")
            or ""
        )
        content = str(content).strip()
        if not content:
            continue

        source = normalize_source(str(item.get("source") or item.get("origin") or item.get("source_type") or "unknown"))
        timestamp = normalize_date(item.get("timestamp") or item.get("date") or item.get("created_at"))
        title = str(item.get("title") or content[:80] or f"Document {idx + 1}")

        normalized.append({
            "id": str(item.get("id", idx)),
            "title": title,
            "content": content,
            "text": content,
            "source": source,
            "source_type": source,
            "author": item.get("author") or item.get("username") or item.get("handle") or "",
            "timestamp": timestamp,
            "date": timestamp,
            "url": item.get("url") or "",
            "language": item.get("language") or "en",
            "category": item.get("category") or source,
        })
    return normalized


def split_for_legacy_scripts(items: List[Dict[str, Any]], work_dir: Path) -> Tuple[str, str]:
    articles = []
    tweets = []

    for item in items:
        if item["source"] == "press":
            articles.append({
                "title": item["title"],
                "text": item["content"],
                "source": item.get("author") or "Press",
                "date": item["timestamp"],
                "url": item.get("url", ""),
            })
        else:
            try:
                year_month = datetime.fromisoformat(item["timestamp"][:10]).strftime("%Y-%m")
            except ValueError:
                year_month = datetime.now().strftime("%Y-%m")
            tweets.append({
                "text": item["content"],
                "handle": item.get("author", ""),
                "date": item["timestamp"],
                "year_month": year_month,
                "url": item.get("url", ""),
            })

    press_path = work_dir / "press.json"
    twitter_path = work_dir / "twitter.json"
    press_path.write_text(json.dumps({"articles": articles}, ensure_ascii=False), encoding="utf-8")
    twitter_path.write_text(json.dumps({"tweets": tweets}, ensure_ascii=False), encoding="utf-8")
    return str(press_path), str(twitter_path)


def safe_run(name: str, fn):
    try:
        return {"status": "completed", "data": fn()}
    except Exception as exc:
        log(f"{name} failed: {exc}")
        return {"status": "failed", "error": str(exc)}


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    output_dir = Path(args.output_dir or dataset_path.parent / f"advanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = read_json(str(dataset_path))
    items = normalize_items(unwrap_items(raw))
    log(f"Loaded {len(items)} normalized documents.")

    if not items:
        raise ValueError("No analyzable documents found in JSON.")

    with tempfile.TemporaryDirectory(prefix="insightflow_advanced_") as tmp:
        tmp_path = Path(tmp)
        press_path, twitter_path = split_for_legacy_scripts(items, tmp_path)

        results: Dict[str, Any] = {
            "generated_at": datetime.now().isoformat(),
            "dataset": str(dataset_path),
            "output_dir": str(output_dir),
            "document_count": len(items),
            "source_distribution": {},
            "modules": {},
        }

        for item in items:
            source = item["source"]
            results["source_distribution"][source] = results["source_distribution"].get(source, 0) + 1

        def run_linguistic():
            from linguistic_metrics import run_linguistic_analysis
            corpus = [
                {
                    "id": item["id"],
                    "source": item["source"],
                    "platform": item.get("source_type", ""),
                    "language": item.get("language", "en"),
                    "category": item.get("category", ""),
                    "date": item["timestamp"][:7],
                    "title": item["title"],
                    "text": item["content"],
                }
                for item in items
            ]
            return run_linguistic_analysis(corpus, output_dir=str(output_dir / "linguistic"), show_plots=False)

        def run_emotional():
            from emotional_intensity import run_emotional_analysis
            return run_emotional_analysis(
                press_path=press_path,
                twitter_path=twitter_path,
                linguistic_json=str(output_dir / "linguistic" / "linguistic_metrics_results.json"),
                output_dir=str(output_dir / "emotional"),
                show_plots=False,
            )

        def run_topics():
            from bertopic_analysis_new import run_topic_analysis
            return run_topic_analysis(items, output_dir=str(output_dir / "topics"))

        results["modules"]["linguistic"] = safe_run("linguistic_metrics", run_linguistic)
        results["modules"]["emotional"] = safe_run("emotional_intensity", run_emotional)
        results["modules"]["topics"] = safe_run("bertopic_analysis_new", run_topics)

    summary_path = output_dir / "advanced_analysis_summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("---ADVANCED_ANALYSIS_START---")
    print(json.dumps(results))
    print("---ADVANCED_ANALYSIS_END---")
    log("Advanced analysis complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"Fatal error: {exc}")
        sys.exit(1)
