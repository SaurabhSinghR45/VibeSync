# 🎨 VibeSync AI — Frontend Architecture

The **VibeSync AI** frontend is a high-performance **Glassmorphic Single Page Application (SPA)** designed for real-time music intelligence, semantic search exploration, and quantitative market analytics.

---

## 📂 File Layout

```
frontend/
├── index.html       # Primary SPA template (HTML5, Tailwind CSS, Space Grotesk & Manrope)
└── README.md        # Frontend documentation
```

---

## 💎 Design System & Features

- **Styling & Tokens**: Custom dark mode palette (`#0e0e13` background, `#c59aff` primary, `#00eefc` secondary cyan, `#ff6f7c` tertiary accent).
- **Glassmorphism**: Backdrop blur overlays with subtle border lighting (`backdrop-filter: blur(24px)`).
- **Interactive Audio Visualizer**: HTML5 Canvas audio spectrum analyzer wired to the persistent Web Audio player.
- **50/30/20 Quantitative Score Badges**: Live breakdown on recommended cards:
  - 🎯 **Semantic Similarity** (50% weight)
  - 🎭 **Emotion Alignment** (30% weight)
  - ⚡ **Acoustic Proximity** (20% weight)
  - 🏷️ **Acoustic Cluster ID**
- **Dual-Pane Lyric Translation Reader**: Line-by-line emotion sentiment heatmaps (Hindi, Punjabi, Spanish to English).
- **Market Analytics Dashboard**: Modal powered by SQL CTEs displaying decadal emotion trends and correlation metrics with CSV export.

---

## 🔌 API Integration

The frontend communicates with the backend via REST JSON endpoints:
- `POST /api/v1/recommend`: Hybrid vector search & XAI recommendations.
- `POST /api/v1/lyrics/translate`: Dual-pane lyric translation & emotion heatmaps.
- `GET /api/v1/analytics/trends`: Decadal mood progression.
- `GET /api/v1/analytics/correlations`: Sentiment-acoustic correlation index.
- `GET /api/v1/analytics/export-csv`: Tabular analytics download.
- `GET /api/v1/history` & `GET /api/v1/likes`: User state and bookmarks.
