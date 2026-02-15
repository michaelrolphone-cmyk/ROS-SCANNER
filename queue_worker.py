#!/usr/bin/env python3
"""
Job queue worker for ros_ultra_ocr.py.

Polls a SQLite database for pending OCR jobs, processes them via
ros_ultra_ocr.py, and stores the extracted metadata back in the database.

Usage:
    python queue_worker.py [--db queue.db] [--poll-interval 5] [--quality max]
"""

import argparse
import json
import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "queue.db")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OCR_SCRIPT = os.path.join(SCRIPT_DIR, "ros_ultra_ocr.py")

shutdown_requested = False


def handle_signal(signum, frame):
    global shutdown_requested
    shutdown_requested = True
    print(f"\n[worker] Received signal {signum}, finishing current job then exiting...")


def init_db(db_path):
    """Initialize the SQLite database schema if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS jobs (
            id          TEXT PRIMARY KEY,
            filename    TEXT NOT NULL,
            filepath    TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'pending',
            quality     TEXT NOT NULL DEFAULT 'max',
            created_at  TEXT NOT NULL,
            started_at  TEXT,
            finished_at TEXT,
            error       TEXT,
            result      TEXT,
            progress    TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
        CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);
    """)
    conn.commit()
    conn.close()


def claim_next_job(db_path):
    """Atomically claim the oldest pending job. Returns job dict or None."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    now = datetime.now(timezone.utc).isoformat()
    try:
        cur = conn.execute("""
            UPDATE jobs
            SET status = 'processing', started_at = ?, progress = 'Starting OCR...'
            WHERE id = (
                SELECT id FROM jobs WHERE status = 'pending'
                ORDER BY created_at ASC LIMIT 1
            )
            RETURNING *
        """, (now,))
        row = cur.fetchone()
        conn.commit()
        if row:
            return dict(row)
        return None
    finally:
        conn.close()


def update_job_progress(db_path, job_id, progress_msg):
    """Update the progress message for a running job."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("UPDATE jobs SET progress = ? WHERE id = ?", (progress_msg, job_id))
    conn.commit()
    conn.close()


def complete_job(db_path, job_id, result_json):
    """Mark a job as completed with its result."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout=5000")
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        UPDATE jobs SET status = 'completed', finished_at = ?, result = ?, progress = NULL
        WHERE id = ?
    """, (now, json.dumps(result_json), job_id))
    conn.commit()
    conn.close()


def fail_job(db_path, job_id, error_msg):
    """Mark a job as failed with an error message."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout=5000")
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        UPDATE jobs SET status = 'failed', finished_at = ?, error = ?, progress = NULL
        WHERE id = ?
    """, (now, error_msg, job_id))
    conn.commit()
    conn.close()


def process_job(job, db_path):
    """Run ros_ultra_ocr.py on the job's PDF and store the result."""
    job_id = job["id"]
    filepath = job["filepath"]
    quality = job["quality"]

    if not os.path.isfile(filepath):
        fail_job(db_path, job_id, f"PDF file not found: {filepath}")
        return

    update_job_progress(db_path, job_id, f"Running OCR ({quality} quality)...")

    # Create a temp file for the JSON output
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
            fail_job(db_path, job_id, f"OCR process exited with code {proc.returncode}: {stderr[-2000:]}")
            return

        # Read the output JSON
        if not os.path.isfile(output_path):
            fail_job(db_path, job_id, "OCR process did not produce output file")
            return

        with open(output_path, "r") as f:
            result = json.load(f)

        complete_job(db_path, job_id, result)
        print(f"[worker] Job {job_id} completed: {result.get('count', 0)} pairs extracted")

    except json.JSONDecodeError as e:
        fail_job(db_path, job_id, f"Failed to parse OCR output JSON: {e}")
    except Exception as e:
        fail_job(db_path, job_id, f"Unexpected error: {traceback.format_exc()[-2000:]}")
    finally:
        if os.path.isfile(output_path):
            os.unlink(output_path)


def queue_stats(db_path):
    """Print current queue statistics."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout=5000")
    cur = conn.execute("""
        SELECT status, COUNT(*) as cnt FROM jobs GROUP BY status
    """)
    stats = {row[0]: row[1] for row in cur.fetchall()}
    conn.close()
    parts = [f"{s}={c}" for s, c in sorted(stats.items())]
    return ", ".join(parts) if parts else "empty"


def main():
    parser = argparse.ArgumentParser(description="OCR job queue worker")
    parser.add_argument("--db", default=DB_PATH, help="Path to SQLite database")
    parser.add_argument("--poll-interval", type=float, default=5.0,
                        help="Seconds between polling for new jobs")
    parser.add_argument("--quality", default=None,
                        help="Override quality for all jobs (fast/balanced/max)")
    parser.add_argument("--once", action="store_true",
                        help="Process one job and exit (for testing)")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    init_db(args.db)
    print(f"[worker] Started. DB: {args.db}, poll interval: {args.poll_interval}s")
    print(f"[worker] Queue: {queue_stats(args.db)}")

    while not shutdown_requested:
        job = claim_next_job(args.db)
        if job:
            if args.quality:
                job["quality"] = args.quality
            print(f"[worker] Processing job {job['id']}: {job['filename']} ({job['quality']})")
            process_job(job, args.db)
            print(f"[worker] Queue: {queue_stats(args.db)}")
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
