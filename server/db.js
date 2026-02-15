const Database = require("better-sqlite3");
const path = require("path");

const DB_PATH = process.env.DB_PATH || path.join(__dirname, "..", "queue.db");

let db;

function getDb() {
  if (!db) {
    db = new Database(DB_PATH);
    db.pragma("journal_mode = WAL");
    db.pragma("busy_timeout = 5000");
    db.exec(`
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
    `);
  }
  return db;
}

function createJob({ id, filename, filepath, quality }) {
  const db = getDb();
  const now = new Date().toISOString();
  db.prepare(`
    INSERT INTO jobs (id, filename, filepath, status, quality, created_at)
    VALUES (?, ?, ?, 'pending', ?, ?)
  `).run(id, filename, filepath, quality, now);
  return getJob(id);
}

function getJob(id) {
  const db = getDb();
  const row = db.prepare("SELECT * FROM jobs WHERE id = ?").get(id);
  if (!row) return null;
  return formatJob(row);
}

function listJobs({ status, limit = 100, offset = 0 } = {}) {
  const db = getDb();
  let sql = "SELECT * FROM jobs";
  const params = [];
  if (status) {
    sql += " WHERE status = ?";
    params.push(status);
  }
  sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?";
  params.push(limit, offset);
  const rows = db.prepare(sql).all(...params);
  return rows.map(formatJob);
}

function deleteJob(id) {
  const db = getDb();
  const job = db.prepare("SELECT * FROM jobs WHERE id = ?").get(id);
  if (!job) return null;
  db.prepare("DELETE FROM jobs WHERE id = ?").run(id);
  return formatJob(job);
}

function getQueueStats() {
  const db = getDb();
  const rows = db.prepare("SELECT status, COUNT(*) as count FROM jobs GROUP BY status").all();
  const stats = { pending: 0, processing: 0, completed: 0, failed: 0 };
  for (const row of rows) {
    stats[row.status] = row.count;
  }
  stats.total = Object.values(stats).reduce((a, b) => a + b, 0);
  return stats;
}

function formatJob(row) {
  const job = { ...row };
  // Parse result JSON if present
  if (job.result) {
    try {
      job.result = JSON.parse(job.result);
    } catch {
      // leave as string if not valid JSON
    }
  }
  return job;
}

function close() {
  if (db) {
    db.close();
    db = null;
  }
}

module.exports = { getDb, createJob, getJob, listJobs, deleteJob, getQueueStats, close };
