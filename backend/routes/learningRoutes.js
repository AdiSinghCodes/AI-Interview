const router = require("express").Router();

const {
  getProfileContext,
  searchInternet,
  searchFiles,
  recommendCourses,
  askAI,
} = require("../controllers/learningController");

const { requireAuth } = require("../middleware/authMiddleware");

// ============================================================
// PROFILE
// ============================================================

router.get(
  "/profile",
  requireAuth,
  getProfileContext
);

// ============================================================
// PERSONALIZED COURSES
// ============================================================

router.get(
  "/courses",
  requireAuth,
  recommendCourses
);

// ============================================================
// INTERNET COURSE / RESOURCE SEARCH
// ============================================================

// Frontend uses:
// /api/learning/internet-courses?query=...

router.get(
  "/internet-courses",
  requireAuth,
  searchInternet
);

// Keep /internet as an alias as well.
router.get(
  "/internet",
  requireAuth,
  searchInternet
);

// ============================================================
// USER FILE SEARCH
// ============================================================

router.get(
  "/files",
  requireAuth,
  searchFiles
);

// ============================================================
// ASK YOUR AI
// ============================================================

router.post(
  "/ask",
  requireAuth,
  askAI
);

module.exports = router;