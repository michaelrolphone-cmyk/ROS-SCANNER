#!/usr/bin/env python3
"""
Job queue worker for ros_ultra_ocr.py.

Polls a jobs directory for pending OCR jobs (JSON files), processes them
via ros_ultra_ocr.py, and writes the extracted metadata back to the job file.

Usage:
    python queue_worker.py [--jobs-dir jobs] [--poll-interval 5] [--quality max]
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(SCRIPT_DIR, "jobs")
OCR_SCRIPT = os.path.join(SCRIPT_DIR, "ros_ultra_ocr.py")

shutdown_requested = False


def handle_signal(signum, frame):
    global shutdown_requested
    shutdown_requested = True
    print(f"\n[worker] Received signal {signum}, finishing current job then exiting...")


def read_job(job_path):
    """Read and parse a job JSON file."""
    with open(job_path, "r") as f:
        return json.load(f)


def write_job(job_path, job):
    """Atomically write a job JSON file (write to tmp then rename)."""
    tmp_path = job_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(job, f, indent=2)
    os.replace(tmp_path, job_path)


def claim_next_job(jobs_dir):
    """Find the oldest pending job, atomically claim it. Returns (path, job) or (None, None)."""
    try:
        files = sorted(Path(jobs_dir).glob("*.json"))
    except FileNotFoundError:
        return None, None

    # Sort by created_at to get oldest first
    pending = []
    for f in files:
        if f.name.endswith(".tmp"):
            continue
        try:
            job = read_job(str(f))
            if job.get("status") == "pending":
                pending.append((str(f), job))
        except (json.JSONDecodeError, OSError):
            continue

    if not pending:
        return None, None

    # Sort by created_at ascending
    pending.sort(key=lambda x: x[1].get("created_at", ""))

    job_path, job = pending[0]
    now = datetime.now(timezone.utc).isoformat()
    job["status"] = "processing"
    job["started_at"] = now
    job["progress"] = "Starting OCR..."
    write_job(job_path, job)
    return job_path, job


def update_job(job_path, updates):
    """Read a job, apply updates, write it back."""
    job = read_job(job_path)
    job.update(updates)
    write_job(job_path, job)
    return job


def process_job(job_path, job):
    """Run ros_ultra_ocr.py on the job's PDF and store the result."""
    job_id = job["id"]
    filepath = job["filepath"]
    quality = job["quality"]

    if not os.path.isfile(filepath):
        now = datetime.now(timezone.utc).isoformat()
        update_job(job_path, {
            "status": "failed",
            "finished_at": now,
            "error": f"PDF file not found: {filepath}",
            "progress": None,
        })
        return

    update_job(job_path, {"progress": f"Running OCR ({quality} quality)..."})

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        output_path = tmp.name

    try:
        cmd = [
            sys.executable, OCR_SCRIPT,
            filepath,
            "--quality", quality,
            "--output", output_path,
        ]

        print(f"[worker] Running: {' '.join(cmd)}")
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=SCRIPT_DIR,
        )

        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            now = datetime.now(timezone.utc).isoformat()
            update_job(job_path, {
                "status": "failed",
                "finished_at": now,
                "error": f"OCR process exited with code {proc.returncode}: {stderr[-2000:]}",
                "progress": None,
            })
            return

        if not os.path.isfile(output_path):
            now = datetime.now(timezone.utc).isoformat()
            update_job(job_path, {
                "status": "failed",
                "finished_at": now,
                "error": "OCR process did not produce output file",
                "progress": None,
            })
            return

        with open(output_path, "r") as f:
            result = json.load(f)

        now = datetime.now(timezone.utc).isoformat()
        update_job(job_path, {
            "status": "completed",
            "finished_at": now,
            "result": result,
            "progress": None,
        })
        print(f"[worker] Job {job_id} completed: {result.get('count', 0)} pairs extracted")

    except json.JSONDecodeError as e:
        now = datetime.now(timezone.utc).isoformat()
        update_job(job_path, {
            "status": "failed",
            "finished_at": now,
            "error": f"Failed to parse OCR output JSON: {e}",
            "progress": None,
        })
    except Exception:
        now = datetime.now(timezone.utc).isoformat()
        update_job(job_path, {
            "status": "failed",
            "finished_at": now,
            "error": f"Unexpected error: {traceback.format_exc()[-2000:]}",
            "progress": None,
        })
    finally:
        if os.path.isfile(output_path):
            os.unlink(output_path)


def queue_stats(jobs_dir):
    """Return current queue statistics as a string."""
    stats = {}
    try:
        for f in Path(jobs_dir).glob("*.json"):
            if f.name.endswith(".tmp"):
                continue
            try:
                job = read_job(str(f))
                s = job.get("status", "unknown")
                stats[s] = stats.get(s, 0) + 1
            except (json.JSONDecodeError, OSError):
                continue
    except FileNotFoundError:
        pass
    parts = [f"{s}={c}" for s, c in sorted(stats.items())]
    return ", ".join(parts) if parts else "empty"


def main():
    parser = argparse.ArgumentParser(description="OCR job queue worker")
    parser.add_argument("--jobs-dir", default=JOBS_DIR, help="Path to jobs directory")
    parser.add_argument("--poll-interval", type=float, default=5.0,
                        help="Seconds between polling for new jobs")
    parser.add_argument("--quality", default=None,
                        help="Override quality for all jobs (fast/balanced/max)")
    parser.add_argument("--once", action="store_true",
                        help="Process one job and exit (for testing)")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    os.makedirs(args.jobs_dir, exist_ok=True)
    print(f"[worker] Started. Jobs dir: {args.jobs_dir}, poll interval: {args.poll_interval}s")
    print(f"[worker] Queue: {queue_stats(args.jobs_dir)}")

    while not shutdown_requested:
        job_path, job = claim_next_job(args.jobs_dir)
        if job:
            if args.quality:
                job["quality"] = args.quality
            print(f"[worker] Processing job {job['id']}: {job['filename']} ({job['quality']})")
            process_job(job_path, job)
            print(f"[worker] Queue: {queue_stats(args.jobs_dir)}")
            if args.once:
                break
        else:
            if args.once:
                print("[worker] No pending jobs.")
                break
            time.sleep(args.poll_interval)

    print("[worker] Shut down.")


if __name__ == "__main__":
    main()
