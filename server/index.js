const express = require("express");
const multer = require("multer");
const { v4: uuidv4 } = require("uuid");
const path = require("path");
const fs = require("fs");
const { createJob, getJob, listJobs, deleteJob, getQueueStats, close } = require("./db");

const app = express();
const PORT = process.env.PORT || 3000;
const UPLOAD_DIR = process.env.UPLOAD_DIR || path.join(__dirname, "..", "uploads");

// Ensure upload directory exists
fs.mkdirSync(UPLOAD_DIR, { recursive: true });

// Configure multer for PDF uploads
const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, UPLOAD_DIR),
  filename: (_req, file, cb) => {
    const id = uuidv4();
    const ext = path.extname(file.originalname) || ".pdf";
    cb(null, `${id}${ext}`);
  },
});

const upload = multer({
  storage,
  fileFilter: (_req, file, cb) => {
    if (file.mimetype === "application/pdf" || path.extname(file.originalname).toLowerCase() === ".pdf") {
      cb(null, true);
    } else {
      cb(new Error("Only PDF files are accepted"));
    }
  },
  limits: { fileSize: 500 * 1024 * 1024 }, // 500MB max
});

app.use(express.json());

// ─── Health / Stats ───────────────────────────────────────────────
app.get("/api/health", (_req, res) => {
  res.json({ status: "ok", timestamp: new Date().toISOString() });
});

app.get("/api/stats", (_req, res) => {
  const stats = getQueueStats();
  res.json(stats);
});

// ─── Submit PDFs ──────────────────────────────────────────────────
// POST /api/jobs  — upload one or more PDFs
app.post("/api/jobs", upload.array("files", 50), (req, res) => {
  if (!req.files || req.files.length === 0) {
    return res.status(400).json({ error: "No PDF files uploaded. Use form field 'files'." });
  }

  const quality = req.body.quality || "max";
  if (!["fast", "balanced", "max"].includes(quality)) {
    return res.status(400).json({ error: "quality must be one of: fast, balanced, max" });
  }

  const jobs = [];
  for (const file of req.files) {
    const id = path.basename(file.filename, path.extname(file.filename));
    const job = createJob({
      id,
      filename: file.originalname,
      filepath: file.path,
      quality,
    });
    jobs.push(job);
  }

  res.status(201).json({
    message: `${jobs.length} job(s) queued for processing`,
    jobs: jobs.map(summarizeJob),
  });
});

// ─── List Jobs ────────────────────────────────────────────────────
// GET /api/jobs?status=pending|processing|completed|failed&limit=100&offset=0
app.get("/api/jobs", (req, res) => {
  const { status, limit, offset } = req.query;
  const jobs = listJobs({
    status: status || undefined,
    limit: limit ? parseInt(limit, 10) : 100,
    offset: offset ? parseInt(offset, 10) : 0,
  });
  res.json({
    count: jobs.length,
    jobs: jobs.map(summarizeJob),
  });
});

// ─── Get Job Detail ───────────────────────────────────────────────
// GET /api/jobs/:id
app.get("/api/jobs/:id", (req, res) => {
  const job = getJob(req.params.id);
  if (!job) return res.status(404).json({ error: "Job not found" });
  res.json(job);
});

// ─── Get Job Result (extracted metadata only) ─────────────────────
// GET /api/jobs/:id/result
app.get("/api/jobs/:id/result", (req, res) => {
  const job = getJob(req.params.id);
  if (!job) return res.status(404).json({ error: "Job not found" });
  if (job.status !== "completed") {
    return res.status(409).json({
      error: "Job not yet completed",
      status: job.status,
      progress: job.progress,
    });
  }
  res.json(job.result);
});

// ─── Delete Job ───────────────────────────────────────────────────
// DELETE /api/jobs/:id
app.delete("/api/jobs/:id", (req, res) => {
  const job = deleteJob(req.params.id);
  if (!job) return res.status(404).json({ error: "Job not found" });

  // Clean up uploaded file
  if (job.filepath && fs.existsSync(job.filepath)) {
    fs.unlinkSync(job.filepath);
  }

  res.json({ message: "Job deleted", id: req.params.id });
});

// ─── List Completed Jobs ──────────────────────────────────────────
// GET /api/completed  — convenience endpoint
app.get("/api/completed", (req, res) => {
  const { limit, offset } = req.query;
  const jobs = listJobs({
    status: "completed",
    limit: limit ? parseInt(limit, 10) : 100,
    offset: offset ? parseInt(offset, 10) : 0,
  });
  res.json({
    count: jobs.length,
    jobs,
  });
});

// ─── Error handler ────────────────────────────────────────────────
app.use((err, _req, res, _next) => {
  if (err instanceof multer.MulterError) {
    return res.status(400).json({ error: err.message });
  }
  if (err.message === "Only PDF files are accepted") {
    return res.status(400).json({ error: err.message });
  }
  console.error("[server] Unhandled error:", err);
  res.status(500).json({ error: "Internal server error" });
});

// ─── Start ────────────────────────────────────────────────────────
const server = app.listen(PORT, () => {
  console.log(`[server] ROS-SCANNER API running on http://localhost:${PORT}`);
  console.log(`[server] Upload dir: ${UPLOAD_DIR}`);
  console.log(`[server] Queue stats:`, getQueueStats());
});

// Graceful shutdown
process.on("SIGINT", () => {
  console.log("\n[server] Shutting down...");
  close();
  server.close(() => process.exit(0));
});
process.on("SIGTERM", () => {
  close();
  server.close(() => process.exit(0));
});

function summarizeJob(job) {
  return {
    id: job.id,
    filename: job.filename,
    status: job.status,
    quality: job.quality,
    created_at: job.created_at,
    started_at: job.started_at,
    finished_at: job.finished_at,
    progress: job.progress,
    error: job.error,
    pair_count: job.result ? job.result.count || 0 : null,
  };
}

module.exports = app;
