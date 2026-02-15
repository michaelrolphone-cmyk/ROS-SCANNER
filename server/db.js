const fs = require("fs");
const path = require("path");

const JOBS_DIR = process.env.JOBS_DIR || path.join(__dirname, "..", "jobs");

// Ensure jobs directory exists
fs.mkdirSync(JOBS_DIR, { recursive: true });

function jobPath(id) {
  return path.join(JOBS_DIR, `${id}.json`);
}

function readJob(id) {
  const p = jobPath(id);
  if (!fs.existsSync(p)) return null;
  try {
    return JSON.parse(fs.readFileSync(p, "utf8"));
  } catch {
    return null;
  }
}

function writeJob(job) {
  const p = jobPath(job.id);
  const tmp = p + ".tmp";
  fs.writeFileSync(tmp, JSON.stringify(job, null, 2));
  fs.renameSync(tmp, p);
}

function createJob({ id, filename, filepath, quality }) {
  const job = {
    id,
    filename,
    filepath,
    status: "pending",
    quality,
    created_at: new Date().toISOString(),
    started_at: null,
    finished_at: null,
    error: null,
    result: null,
    progress: null,
  };
  writeJob(job);
  return job;
}

function getJob(id) {
  return readJob(id);
}

function listJobs({ status, limit = 100, offset = 0 } = {}) {
  let files;
  try {
    files = fs.readdirSync(JOBS_DIR).filter((f) => f.endsWith(".json") && !f.endsWith(".tmp"));
  } catch {
    return [];
  }

  let jobs = [];
  for (const f of files) {
    try {
      const job = JSON.parse(fs.readFileSync(path.join(JOBS_DIR, f), "utf8"));
      if (!status || job.status === status) {
        jobs.push(job);
      }
    } catch {
      continue;
    }
  }

  // Sort by created_at descending (newest first)
  jobs.sort((a, b) => (b.created_at || "").localeCompare(a.created_at || ""));
  return jobs.slice(offset, offset + limit);
}

function deleteJob(id) {
  const job = readJob(id);
  if (!job) return null;
  const p = jobPath(id);
  try {
    fs.unlinkSync(p);
  } catch {
    // already gone
  }
  return job;
}

function getQueueStats() {
  const jobs = listJobs({ limit: Infinity });
  const stats = { pending: 0, processing: 0, completed: 0, failed: 0 };
  for (const job of jobs) {
    const s = job.status || "unknown";
    if (s in stats) stats[s]++;
  }
  stats.total = Object.values(stats).reduce((a, b) => a + b, 0);
  return stats;
}

module.exports = { createJob, getJob, listJobs, deleteJob, getQueueStats };
