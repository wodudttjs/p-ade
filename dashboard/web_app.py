#!/usr/bin/env python3
"""
P-ADE 웹 대시보드

Flask + Plotly Dash 기반 웹 대시보드
- 실시간 파이프라인 모니터링
- DB 통계 시각화
- 파이프라인 제어 (Start/Stop)
"""

import os
import sys
import json
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# DataService import
try:
    from dashboard.data_service import get_data_service
    HAS_DATA_SERVICE = True
except ImportError:
    HAS_DATA_SERVICE = False

# ============================================================================
# Flask App
# ============================================================================

app = Flask(__name__)
CORS(app)

# 전역 상태
pipeline_state = {
    "is_running": False,
    "current_stage": None,
    "progress": {
        "crawl": 0,
        "download": 0,
        "detect": 0,
        "upload": 0,
    },
    "logs": [],
    "started_at": None,
    "process": None,
}

data_service = None
if HAS_DATA_SERVICE:
    try:
        data_service = get_data_service()
    except Exception:
        pass


def get_file_stats() -> Dict[str, int]:
    """파일 기반 통계"""
    data_dir = PROJECT_ROOT / "data"
    stats = {
        "raw_videos": 0,
        "episodes": 0,
        "poses": 0,
        "total_size_mb": 0,
    }
    
    try:
        raw_dir = data_dir / "raw"
        if raw_dir.exists():
            mp4_files = list(raw_dir.glob("*.mp4"))
            stats["raw_videos"] = len(mp4_files)
            stats["total_size_mb"] += sum(f.stat().st_size for f in mp4_files) / (1024 * 1024)
        
        episodes_dir = data_dir / "episodes"
        if episodes_dir.exists():
            npz_files = list(episodes_dir.glob("*.npz"))
            stats["episodes"] = len(npz_files)
            stats["total_size_mb"] += sum(f.stat().st_size for f in npz_files) / (1024 * 1024)
        
        poses_dir = data_dir / "poses"
        if poses_dir.exists():
            stats["poses"] = len(list(poses_dir.glob("*.npz")))
    except Exception:
        pass
    
    return stats


def get_db_stats() -> Dict[str, Any]:
    """DB 통계"""
    if not data_service:
        return {"connected": False}
    
    try:
        kpi = data_service.get_kpi()
        return {
            "connected": True,
            "total_videos": kpi.total_videos,
            "episodes": kpi.episodes,
            "queue_depth": kpi.queue_depth,
            "storage_gb": kpi.storage_gb,
            "avg_quality": kpi.avg_quality_score,
        }
    except Exception as e:
        return {"connected": False, "error": str(e)}


# ============================================================================
# HTML Template
# ============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>P-ADE Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #0d1117;
            --bg-card: #161b22;
            --bg-hover: #21262d;
            --border-color: #30363d;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --accent-blue: #58a6ff;
            --accent-green: #3fb950;
            --accent-yellow: #d29922;
            --accent-red: #f85149;
            --accent-purple: #a371f7;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            min-height: 100vh;
        }
        
        /* Sidebar */
        .sidebar {
            position: fixed;
            left: 0;
            top: 0;
            width: 220px;
            height: 100vh;
            background: var(--bg-card);
            border-right: 1px solid var(--border-color);
            padding: 20px 0;
            z-index: 1000;
        }
        
        .sidebar-logo {
            padding: 0 20px 20px;
            font-size: 24px;
            font-weight: 800;
            color: var(--accent-blue);
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 20px;
        }
        
        .sidebar-nav {
            list-style: none;
        }
        
        .sidebar-nav li a {
            display: flex;
            align-items: center;
            padding: 12px 20px;
            color: var(--text-secondary);
            text-decoration: none;
            transition: all 0.2s;
        }
        
        .sidebar-nav li a:hover,
        .sidebar-nav li a.active {
            background: var(--bg-hover);
            color: var(--text-primary);
            border-left: 3px solid var(--accent-blue);
        }
        
        .sidebar-nav li a i {
            margin-right: 12px;
            font-size: 18px;
        }
        
        .sidebar-footer {
            position: absolute;
            bottom: 20px;
            left: 0;
            right: 0;
            padding: 0 20px;
        }
        
        .db-status {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 10px;
            background: var(--bg-dark);
            border-radius: 8px;
            font-size: 12px;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent-red);
        }
        
        .status-dot.connected {
            background: var(--accent-green);
        }
        
        /* Main Content */
        .main-content {
            margin-left: 220px;
            min-height: 100vh;
        }
        
        /* Top Bar */
        .top-bar {
            background: var(--bg-card);
            border-bottom: 1px solid var(--border-color);
            padding: 15px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .page-title {
            font-size: 20px;
            font-weight: 600;
        }
        
        .top-actions {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        
        .last-update {
            color: var(--text-secondary);
            font-size: 13px;
        }
        
        .btn-icon {
            width: 36px;
            height: 36px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            background: var(--bg-dark);
            color: var(--text-primary);
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .btn-icon:hover {
            background: var(--bg-hover);
            border-color: var(--accent-blue);
        }
        
        /* Control Panel */
        .control-panel {
            background: var(--bg-card);
            border-bottom: 1px solid var(--border-color);
            padding: 20px 30px;
        }
        
        .control-grid {
            display: grid;
            grid-template-columns: 220px 1fr 220px;
            gap: 20px;
        }
        
        .control-box {
            background: var(--bg-dark);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px;
        }
        
        .control-box h4 {
            font-size: 13px;
            color: var(--text-secondary);
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .btn-start {
            width: 100%;
            padding: 12px;
            border: none;
            border-radius: 8px;
            background: var(--accent-green);
            color: #fff;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            margin-bottom: 10px;
        }
        
        .btn-start:hover {
            filter: brightness(1.1);
        }
        
        .btn-start:disabled {
            background: var(--text-secondary);
            cursor: not-allowed;
        }
        
        .btn-stop {
            width: 100%;
            padding: 12px;
            border: none;
            border-radius: 8px;
            background: var(--accent-red);
            color: #fff;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .btn-stop:disabled {
            background: var(--text-secondary);
            cursor: not-allowed;
        }
        
        .form-control-dark {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            border-radius: 6px;
            padding: 8px 12px;
        }
        
        .form-control-dark:focus {
            background: var(--bg-card);
            border-color: var(--accent-blue);
            color: var(--text-primary);
            box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.2);
        }
        
        /* Progress Bars */
        .progress-stages {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .stage-item {
            text-align: center;
        }
        
        .stage-label {
            font-size: 12px;
            color: var(--text-secondary);
            margin-bottom: 8px;
        }
        
        .stage-progress {
            height: 8px;
            background: var(--bg-card);
            border-radius: 4px;
            overflow: hidden;
        }
        
        .stage-progress .bar {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }
        
        .stage-progress .bar.crawl { background: var(--accent-blue); }
        .stage-progress .bar.download { background: var(--accent-purple); }
        .stage-progress .bar.detect { background: var(--accent-yellow); }
        .stage-progress .bar.upload { background: var(--accent-green); }
        
        .total-progress {
            height: 12px;
            background: var(--bg-card);
            border-radius: 6px;
            overflow: hidden;
        }
        
        .total-progress .bar {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-blue), var(--accent-green));
            border-radius: 6px;
            transition: width 0.3s ease;
        }
        
        .progress-status {
            display: flex;
            justify-content: space-between;
            margin-top: 8px;
            font-size: 13px;
        }
        
        .progress-status .label {
            color: var(--text-secondary);
        }
        
        .progress-status .value {
            font-weight: 600;
        }
        
        /* Stats */
        .stat-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid var(--border-color);
        }
        
        .stat-row:last-child {
            border-bottom: none;
        }
        
        .stat-label {
            color: var(--text-secondary);
            font-size: 13px;
        }
        
        .stat-value {
            font-weight: 600;
            color: var(--accent-blue);
        }
        
        /* Dashboard Content */
        .dashboard-content {
            padding: 30px;
        }
        
        /* Stats Cards */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
        }
        
        .stat-card .icon {
            width: 48px;
            height: 48px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            margin-bottom: 15px;
        }
        
        .stat-card .icon.blue { background: rgba(88, 166, 255, 0.15); color: var(--accent-blue); }
        .stat-card .icon.green { background: rgba(63, 185, 80, 0.15); color: var(--accent-green); }
        .stat-card .icon.yellow { background: rgba(210, 153, 34, 0.15); color: var(--accent-yellow); }
        .stat-card .icon.purple { background: rgba(163, 113, 247, 0.15); color: var(--accent-purple); }
        
        .stat-card .value {
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 5px;
        }
        
        .stat-card .label {
            color: var(--text-secondary);
            font-size: 14px;
        }
        
        /* Charts Container */
        .charts-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .chart-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
        }
        
        .chart-card h3 {
            font-size: 16px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        /* Recent Activity */
        .activity-list {
            max-height: 300px;
            overflow-y: auto;
        }
        
        .activity-item {
            display: flex;
            gap: 12px;
            padding: 12px 0;
            border-bottom: 1px solid var(--border-color);
        }
        
        .activity-item:last-child {
            border-bottom: none;
        }
        
        .activity-icon {
            width: 32px;
            height: 32px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            flex-shrink: 0;
        }
        
        .activity-icon.success { background: rgba(63, 185, 80, 0.15); color: var(--accent-green); }
        .activity-icon.info { background: rgba(88, 166, 255, 0.15); color: var(--accent-blue); }
        .activity-icon.warning { background: rgba(210, 153, 34, 0.15); color: var(--accent-yellow); }
        
        .activity-content {
            flex: 1;
        }
        
        .activity-title {
            font-size: 13px;
            margin-bottom: 2px;
        }
        
        .activity-time {
            font-size: 11px;
            color: var(--text-secondary);
        }
        
        /* Log Panel */
        .log-panel {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
        }
        
        .log-panel h3 {
            font-size: 16px;
            margin-bottom: 15px;
        }
        
        .log-content {
            background: var(--bg-dark);
            border-radius: 8px;
            padding: 15px;
            max-height: 200px;
            overflow-y: auto;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 12px;
            line-height: 1.6;
        }
        
        .log-line {
            margin-bottom: 4px;
        }
        
        .log-line.info { color: var(--accent-blue); }
        .log-line.success { color: var(--accent-green); }
        .log-line.warning { color: var(--accent-yellow); }
        .log-line.error { color: var(--accent-red); }
        
        /* Responsive */
        @media (max-width: 1200px) {
            .stats-grid {
                grid-template-columns: repeat(2, 1fr);
            }
            
            .charts-grid {
                grid-template-columns: 1fr;
            }
            
            .control-grid {
                grid-template-columns: 1fr;
            }
        }
        
        @media (max-width: 768px) {
            .sidebar {
                transform: translateX(-100%);
            }
            
            .main-content {
                margin-left: 0;
            }
            
            .stats-grid {
                grid-template-columns: 1fr;
            }
            
            .progress-stages {
                grid-template-columns: repeat(2, 1fr);
            }
        }
    </style>
</head>
<body>
    <!-- Sidebar -->
    <aside class="sidebar">
        <div class="sidebar-logo">
            🎬 P-ADE
        </div>
        <ul class="sidebar-nav">
            <li><a href="#" class="active" data-page="overview"><i class="bi bi-graph-up"></i> Overview</a></li>
            <li><a href="#" data-page="jobs"><i class="bi bi-list-task"></i> Jobs</a></li>
            <li><a href="#" data-page="quality"><i class="bi bi-award"></i> Quality</a></li>
            <li><a href="#" data-page="settings"><i class="bi bi-gear"></i> Settings</a></li>
        </ul>
        <div class="sidebar-footer">
            <div class="db-status">
                <span class="status-dot" id="db-status-dot"></span>
                <span id="db-status-text">Checking...</span>
            </div>
            <div style="text-align: center; margin-top: 10px; color: var(--text-secondary); font-size: 11px;">
                v1.0.0
            </div>
        </div>
    </aside>
    
    <!-- Main Content -->
    <main class="main-content">
        <!-- Top Bar -->
        <header class="top-bar">
            <h1 class="page-title">Overview</h1>
            <div class="top-actions">
                <span class="last-update" id="last-update">Last update: —</span>
                <button class="btn-icon" onclick="refreshData()" title="Refresh">
                    <i class="bi bi-arrow-clockwise"></i>
                </button>
                <button class="btn-icon" onclick="toggleTheme()" title="Toggle Theme">
                    <i class="bi bi-moon"></i>
                </button>
            </div>
        </header>
        
        <!-- Control Panel -->
        <section class="control-panel">
            <div class="control-grid">
                <!-- Pipeline Control -->
                <div class="control-box">
                    <h4>Pipeline Control</h4>
                    <button class="btn-start" id="btn-start" onclick="startPipeline()">
                        <i class="bi bi-play-fill"></i> Start Collection
                    </button>
                    <button class="btn-stop" id="btn-stop" onclick="stopPipeline()" disabled>
                        <i class="bi bi-stop-fill"></i> Stop
                    </button>
                    <div style="margin-top: 15px;">
                        <label style="font-size: 12px; color: var(--text-secondary);">Target Videos</label>
                        <input type="number" class="form-control-dark w-100 mt-1" id="target-count" value="50" min="1" max="1000">
                    </div>
                </div>
                
                <!-- Progress -->
                <div class="control-box">
                    <h4>Pipeline Progress</h4>
                    <div class="progress-stages">
                        <div class="stage-item">
                            <div class="stage-label">📡 Crawl</div>
                            <div class="stage-progress">
                                <div class="bar crawl" id="progress-crawl" style="width: 0%"></div>
                            </div>
                        </div>
                        <div class="stage-item">
                            <div class="stage-label">📥 Download</div>
                            <div class="stage-progress">
                                <div class="bar download" id="progress-download" style="width: 0%"></div>
                            </div>
                        </div>
                        <div class="stage-item">
                            <div class="stage-label">🔍 Detect</div>
                            <div class="stage-progress">
                                <div class="bar detect" id="progress-detect" style="width: 0%"></div>
                            </div>
                        </div>
                        <div class="stage-item">
                            <div class="stage-label">☁️ Upload</div>
                            <div class="stage-progress">
                                <div class="bar upload" id="progress-upload" style="width: 0%"></div>
                            </div>
                        </div>
                    </div>
                    <div class="total-progress">
                        <div class="bar" id="progress-total" style="width: 0%"></div>
                    </div>
                    <div class="progress-status">
                        <span class="label">Status:</span>
                        <span class="value" id="pipeline-status">Ready</span>
                    </div>
                </div>
                
                <!-- Database Stats -->
                <div class="control-box">
                    <h4>Database Stats</h4>
                    <div class="stat-row">
                        <span class="stat-label">📹 Videos</span>
                        <span class="stat-value" id="stat-videos">—</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">🎬 Episodes</span>
                        <span class="stat-value" id="stat-episodes">—</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">📋 Queue</span>
                        <span class="stat-value" id="stat-queue">—</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">💾 Storage</span>
                        <span class="stat-value" id="stat-storage">—</span>
                    </div>
                </div>
            </div>
        </section>
        
        <!-- Dashboard Content -->
        <section class="dashboard-content">
            <!-- Stats Cards -->
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="icon blue"><i class="bi bi-film"></i></div>
                    <div class="value" id="card-videos">0</div>
                    <div class="label">Total Videos</div>
                </div>
                <div class="stat-card">
                    <div class="icon green"><i class="bi bi-collection-play"></i></div>
                    <div class="value" id="card-episodes">0</div>
                    <div class="label">Episodes Generated</div>
                </div>
                <div class="stat-card">
                    <div class="icon yellow"><i class="bi bi-hdd"></i></div>
                    <div class="value" id="card-storage">0 MB</div>
                    <div class="label">Storage Used</div>
                </div>
                <div class="stat-card">
                    <div class="icon purple"><i class="bi bi-graph-up-arrow"></i></div>
                    <div class="value" id="card-quality">—</div>
                    <div class="label">Avg Quality Score</div>
                </div>
            </div>
            
            <!-- Charts & Activity -->
            <div class="charts-grid">
                <!-- Pipeline Chart -->
                <div class="chart-card">
                    <h3><i class="bi bi-bar-chart"></i> Pipeline Overview</h3>
                    <div id="pipeline-chart" style="height: 250px; display: flex; align-items: flex-end; gap: 20px; padding: 20px;">
                        <div style="flex: 1; text-align: center;">
                            <div style="background: var(--accent-blue); border-radius: 8px 8px 0 0; transition: height 0.3s;" id="chart-bar-videos"></div>
                            <div style="margin-top: 10px; font-size: 12px; color: var(--text-secondary);">Videos</div>
                        </div>
                        <div style="flex: 1; text-align: center;">
                            <div style="background: var(--accent-purple); border-radius: 8px 8px 0 0; transition: height 0.3s;" id="chart-bar-poses"></div>
                            <div style="margin-top: 10px; font-size: 12px; color: var(--text-secondary);">Poses</div>
                        </div>
                        <div style="flex: 1; text-align: center;">
                            <div style="background: var(--accent-green); border-radius: 8px 8px 0 0; transition: height 0.3s;" id="chart-bar-episodes"></div>
                            <div style="margin-top: 10px; font-size: 12px; color: var(--text-secondary);">Episodes</div>
                        </div>
                    </div>
                </div>
                
                <!-- Recent Activity -->
                <div class="chart-card">
                    <h3><i class="bi bi-clock-history"></i> Recent Activity</h3>
                    <div class="activity-list" id="activity-list">
                        <div class="activity-item">
                            <div class="activity-icon info"><i class="bi bi-info"></i></div>
                            <div class="activity-content">
                                <div class="activity-title">Dashboard started</div>
                                <div class="activity-time">Just now</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Log Panel -->
            <div class="log-panel">
                <h3><i class="bi bi-terminal"></i> Pipeline Logs</h3>
                <div class="log-content" id="log-content">
                    <div class="log-line info">[INFO] Dashboard initialized</div>
                    <div class="log-line">Waiting for pipeline to start...</div>
                </div>
            </div>
        </section>
    </main>
    
    <script>
        // API Base URL
        const API_BASE = '';
        
        // State
        let isRunning = false;
        let refreshInterval = null;
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            refreshData();
            startAutoRefresh();
            setupNavigation();
        });
        
        // Navigation
        function setupNavigation() {
            document.querySelectorAll('.sidebar-nav a').forEach(link => {
                link.addEventListener('click', (e) => {
                    e.preventDefault();
                    document.querySelectorAll('.sidebar-nav a').forEach(l => l.classList.remove('active'));
                    link.classList.add('active');
                    document.querySelector('.page-title').textContent = link.textContent.trim();
                });
            });
        }
        
        // Auto Refresh
        function startAutoRefresh() {
            refreshInterval = setInterval(refreshData, 5000);
        }
        
        // Refresh Data
        async function refreshData() {
            try {
                // Get stats
                const statsRes = await fetch(`${API_BASE}/api/stats`);
                const stats = await statsRes.json();
                
                // Update DB status
                const statusDot = document.getElementById('db-status-dot');
                const statusText = document.getElementById('db-status-text');
                if (stats.db.connected) {
                    statusDot.classList.add('connected');
                    statusText.textContent = 'DB Connected';
                } else {
                    statusDot.classList.remove('connected');
                    statusText.textContent = 'DB Disconnected';
                }
                
                // Update stats
                document.getElementById('stat-videos').textContent = stats.files.raw_videos;
                document.getElementById('stat-episodes').textContent = stats.files.episodes;
                document.getElementById('stat-queue').textContent = stats.db.queue_depth || '—';
                document.getElementById('stat-storage').textContent = `${stats.files.total_size_mb.toFixed(1)} MB`;
                
                // Update cards
                document.getElementById('card-videos').textContent = stats.files.raw_videos;
                document.getElementById('card-episodes').textContent = stats.files.episodes;
                document.getElementById('card-storage').textContent = `${stats.files.total_size_mb.toFixed(1)} MB`;
                document.getElementById('card-quality').textContent = stats.db.avg_quality ? stats.db.avg_quality.toFixed(2) : '—';
                
                // Update chart bars
                const maxVal = Math.max(stats.files.raw_videos, stats.files.poses, stats.files.episodes, 1);
                document.getElementById('chart-bar-videos').style.height = `${(stats.files.raw_videos / maxVal) * 180}px`;
                document.getElementById('chart-bar-poses').style.height = `${(stats.files.poses / maxVal) * 180}px`;
                document.getElementById('chart-bar-episodes').style.height = `${(stats.files.episodes / maxVal) * 180}px`;
                
                // Update timestamp
                const now = new Date().toLocaleTimeString();
                document.getElementById('last-update').textContent = `Last update: ${now}`;
                
                // Get pipeline status
                const pipelineRes = await fetch(`${API_BASE}/api/pipeline/status`);
                const pipeline = await pipelineRes.json();
                
                isRunning = pipeline.is_running;
                updatePipelineUI(pipeline);
                
            } catch (error) {
                console.error('Error refreshing data:', error);
            }
        }
        
        // Update Pipeline UI
        function updatePipelineUI(pipeline) {
            // Buttons
            document.getElementById('btn-start').disabled = pipeline.is_running;
            document.getElementById('btn-stop').disabled = !pipeline.is_running;
            
            // Progress bars
            document.getElementById('progress-crawl').style.width = `${pipeline.progress.crawl}%`;
            document.getElementById('progress-download').style.width = `${pipeline.progress.download}%`;
            document.getElementById('progress-detect').style.width = `${pipeline.progress.detect}%`;
            document.getElementById('progress-upload').style.width = `${pipeline.progress.upload}%`;
            
            // Total progress
            const total = (pipeline.progress.crawl + pipeline.progress.download + 
                         pipeline.progress.detect + pipeline.progress.upload) / 4;
            document.getElementById('progress-total').style.width = `${total}%`;
            
            // Status
            let status = 'Ready';
            if (pipeline.is_running) {
                status = `Running: ${pipeline.current_stage || 'Initializing...'}`;
            } else if (total > 0 && total < 100) {
                status = 'Paused';
            } else if (total >= 100) {
                status = 'Completed';
            }
            document.getElementById('pipeline-status').textContent = status;
            
            // Logs
            if (pipeline.logs && pipeline.logs.length > 0) {
                const logContent = document.getElementById('log-content');
                logContent.innerHTML = pipeline.logs.slice(-20).map(log => {
                    let cls = '';
                    if (log.includes('ERROR') || log.includes('❌')) cls = 'error';
                    else if (log.includes('SUCCESS') || log.includes('✅') || log.includes('완료')) cls = 'success';
                    else if (log.includes('WARN') || log.includes('⚠')) cls = 'warning';
                    else if (log.includes('INFO') || log.includes('🔍') || log.includes('📥')) cls = 'info';
                    return `<div class="log-line ${cls}">${log}</div>`;
                }).join('');
                logContent.scrollTop = logContent.scrollHeight;
            }
        }
        
        // Start Pipeline
        async function startPipeline() {
            const target = document.getElementById('target-count').value;
            
            try {
                addActivity('info', 'Starting pipeline...');
                
                const res = await fetch(`${API_BASE}/api/pipeline/start`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ target_count: parseInt(target) })
                });
                
                const result = await res.json();
                
                if (result.success) {
                    addActivity('success', 'Pipeline started successfully');
                    addLog('[INFO] Pipeline started');
                } else {
                    addActivity('warning', `Failed to start: ${result.message}`);
                }
                
                refreshData();
                
            } catch (error) {
                console.error('Error starting pipeline:', error);
                addActivity('warning', 'Error starting pipeline');
            }
        }
        
        // Stop Pipeline
        async function stopPipeline() {
            try {
                addActivity('warning', 'Stopping pipeline...');
                
                const res = await fetch(`${API_BASE}/api/pipeline/stop`, {
                    method: 'POST'
                });
                
                const result = await res.json();
                
                if (result.success) {
                    addActivity('info', 'Pipeline stopped');
                    addLog('[INFO] Pipeline stopped by user');
                }
                
                refreshData();
                
            } catch (error) {
                console.error('Error stopping pipeline:', error);
            }
        }
        
        // Add Activity
        function addActivity(type, message) {
            const list = document.getElementById('activity-list');
            const icons = {
                success: 'check-lg',
                info: 'info',
                warning: 'exclamation-triangle'
            };
            
            const item = document.createElement('div');
            item.className = 'activity-item';
            item.innerHTML = `
                <div class="activity-icon ${type}"><i class="bi bi-${icons[type]}"></i></div>
                <div class="activity-content">
                    <div class="activity-title">${message}</div>
                    <div class="activity-time">${new Date().toLocaleTimeString()}</div>
                </div>
            `;
            
            list.insertBefore(item, list.firstChild);
            
            // Keep only last 10
            while (list.children.length > 10) {
                list.removeChild(list.lastChild);
            }
        }
        
        // Add Log
        function addLog(message) {
            const logContent = document.getElementById('log-content');
            const line = document.createElement('div');
            line.className = 'log-line';
            if (message.includes('ERROR')) line.classList.add('error');
            else if (message.includes('SUCCESS') || message.includes('완료')) line.classList.add('success');
            else if (message.includes('INFO')) line.classList.add('info');
            line.textContent = message;
            logContent.appendChild(line);
            logContent.scrollTop = logContent.scrollHeight;
        }
        
        // Toggle Theme (placeholder)
        function toggleTheme() {
            document.body.classList.toggle('light-theme');
        }
    </script>
</body>
</html>
"""


# ============================================================================
# API Routes
# ============================================================================

@app.route("/")
def index():
    """메인 페이지"""
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/stats")
def api_stats():
    """통계 API"""
    return jsonify({
        "files": get_file_stats(),
        "db": get_db_stats(),
    })


@app.route("/api/pipeline/status")
def api_pipeline_status():
    """파이프라인 상태"""
    return jsonify({
        "is_running": pipeline_state["is_running"],
        "current_stage": pipeline_state["current_stage"],
        "progress": pipeline_state["progress"],
        "logs": pipeline_state["logs"][-50:],
        "started_at": pipeline_state["started_at"],
    })


@app.route("/api/pipeline/start", methods=["POST"])
def api_pipeline_start():
    """파이프라인 시작"""
    if pipeline_state["is_running"]:
        return jsonify({"success": False, "message": "Pipeline is already running"})
    
    data = request.json or {}
    target_count = data.get("target_count", 50)
    
    def run_pipeline():
        pipeline_state["is_running"] = True
        pipeline_state["started_at"] = datetime.now().isoformat()
        pipeline_state["logs"] = ["[INFO] Pipeline started"]
        
        try:
            # Crawl stage
            pipeline_state["current_stage"] = "crawl"
            pipeline_state["logs"].append(f"[INFO] Starting crawl stage (target: {target_count})")
            
            cmd = [
                sys.executable, str(PROJECT_ROOT / "mass_collector.py"),
                "--target", str(target_count),
                "--stage", "crawl"
            ]
            
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            
            proc = subprocess.Popen(
                cmd, cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", env=env
            )
            pipeline_state["process"] = proc
            
            for line in proc.stdout:
                line = line.strip()
                if line:
                    pipeline_state["logs"].append(line)
                if not pipeline_state["is_running"]:
                    proc.terminate()
                    break
            
            proc.wait()
            pipeline_state["progress"]["crawl"] = 100
            
            if not pipeline_state["is_running"]:
                return
            
            # Download stage
            pipeline_state["current_stage"] = "download"
            pipeline_state["logs"].append("[INFO] Starting download stage")
            
            cmd = [sys.executable, str(PROJECT_ROOT / "mass_collector.py"), "--target", str(target_count), "--stage", "download"]
            proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env)
            pipeline_state["process"] = proc
            
            for line in proc.stdout:
                line = line.strip()
                if line:
                    pipeline_state["logs"].append(line)
                if not pipeline_state["is_running"]:
                    proc.terminate()
                    break
            
            proc.wait()
            pipeline_state["progress"]["download"] = 100
            
            if not pipeline_state["is_running"]:
                return
            
            # Detect stage
            pipeline_state["current_stage"] = "detect"
            pipeline_state["logs"].append("[INFO] Starting detection stage")
            
            cmd = [sys.executable, str(PROJECT_ROOT / "mass_collector.py"), "--target", str(target_count), "--stage", "detect"]
            proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env)
            pipeline_state["process"] = proc
            
            for line in proc.stdout:
                line = line.strip()
                if line:
                    pipeline_state["logs"].append(line)
                if not pipeline_state["is_running"]:
                    proc.terminate()
                    break
            
            proc.wait()
            pipeline_state["progress"]["detect"] = 100
            
            if not pipeline_state["is_running"]:
                return
            
            # Upload stage
            pipeline_state["current_stage"] = "upload"
            pipeline_state["logs"].append("[INFO] Starting upload stage")
            
            cmd = [sys.executable, str(PROJECT_ROOT / "mass_collector.py"), "--target", str(target_count), "--stage", "upload"]
            proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env)
            pipeline_state["process"] = proc
            
            for line in proc.stdout:
                line = line.strip()
                if line:
                    pipeline_state["logs"].append(line)
                if not pipeline_state["is_running"]:
                    proc.terminate()
                    break
            
            proc.wait()
            pipeline_state["progress"]["upload"] = 100
            
            pipeline_state["logs"].append("[SUCCESS] ✅ Pipeline completed!")
            
        except Exception as e:
            pipeline_state["logs"].append(f"[ERROR] {e}")
        
        finally:
            pipeline_state["is_running"] = False
            pipeline_state["current_stage"] = None
            pipeline_state["process"] = None
    
    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()
    
    return jsonify({"success": True, "message": "Pipeline started"})


@app.route("/api/pipeline/stop", methods=["POST"])
def api_pipeline_stop():
    """파이프라인 중지"""
    pipeline_state["is_running"] = False
    
    if pipeline_state["process"]:
        try:
            pipeline_state["process"].terminate()
        except Exception:
            pass
    
    pipeline_state["logs"].append("[INFO] Pipeline stopped by user")
    
    return jsonify({"success": True, "message": "Pipeline stopped"})


# ============================================================================
# Main
# ============================================================================

def run_web_dashboard(host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
    """웹 대시보드 실행"""
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                     P-ADE Web Dashboard                         ║
╠══════════════════════════════════════════════════════════════════╣
║  URL: http://localhost:{port}                                    ║
║  API: http://localhost:{port}/api/stats                          ║
╚══════════════════════════════════════════════════════════════════╝
""")
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="P-ADE Web Dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    
    run_web_dashboard(host=args.host, port=args.port, debug=args.debug)
