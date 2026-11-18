"""
LPU Solar-Glacial UI Design System.
Combines Lovely Professional University (LPU) iconic solar branding (#f37023)
with frosted glacier glassmorphism, aurora cyan glows, and high-tech AI HUD aesthetics.
"""

import base64
from pathlib import Path
import streamlit as st

LOGO_PATH = Path("ui/assets/lpu_round_logo.png")


def get_round_logo_base64() -> str:
    """Returns base64 encoded string of the single round circular LPU logo."""
    if LOGO_PATH.exists():
        try:
            with open(LOGO_PATH, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return ""
    return ""


def apply_lpu_glacial_css():
    """Injects high-end LPU Solar-Glacier frosted glassmorphic CSS styles."""
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

            /* Global Typography & Base */
            html, body, [class*="css"] {
                font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
            }

            .stApp {
                background: linear-gradient(145deg, #060911 0%, #0c121d 40%, #0a1424 70%, #05080e 100%) !important;
                background-attachment: fixed;
            }

            /* Ambient Solar-Glacial Light Refraction */
            .stApp::before {
                content: "";
                position: fixed;
                top: 0; left: 0; right: 0; height: 380px;
                background: radial-gradient(ellipse at 25% -10%, rgba(243, 112, 35, 0.12) 0%, transparent 60%),
                            radial-gradient(ellipse at 75% -10%, rgba(0, 242, 254, 0.10) 0%, transparent 60%);
                pointer-events: none;
                z-index: 0;
            }

            /* Frosted Glacial Glass Card */
            .lpu-card {
                background: rgba(12, 20, 34, 0.72);
                backdrop-filter: blur(24px);
                -webkit-backdrop-filter: blur(24px);
                border: 1px solid rgba(243, 112, 35, 0.22);
                border-radius: 16px;
                padding: 22px 24px;
                box-shadow: 0 10px 35px 0 rgba(0, 0, 0, 0.5), inset 0 1px 1px rgba(255, 255, 255, 0.12);
                transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
                position: relative;
                overflow: hidden;
            }

            .lpu-card:hover {
                transform: translateY(-3px);
                border-color: rgba(243, 112, 35, 0.50);
                box-shadow: 0 14px 44px rgba(243, 112, 35, 0.20), 0 0 20px rgba(0, 242, 254, 0.12);
            }

            .lpu-card::after {
                content: "";
                position: absolute;
                top: -50%; left: -50%; width: 200%; height: 200%;
                background: linear-gradient(45deg, transparent 45%, rgba(255, 255, 255, 0.04) 50%, transparent 55%);
                pointer-events: none;
                transform: rotate(30deg);
            }

            .lpu-kpi-title {
                font-size: 0.80rem;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                color: #a5b4cb;
                font-weight: 600;
                margin-bottom: 8px;
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .lpu-kpi-val {
                font-size: 2.3rem;
                font-weight: 800;
                background: linear-gradient(135deg, #ffffff 10%, #ffd8be 60%, #f37023 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                line-height: 1.1;
                font-family: 'JetBrains Mono', monospace;
            }

            .lpu-kpi-sub {
                font-size: 0.80rem;
                color: #8ed9ec;
                margin-top: 6px;
                font-weight: 500;
            }

            /* LPU Header Showcase Banner */
            .lpu-hero-banner {
                background: linear-gradient(135deg, rgba(243, 112, 35, 0.12) 0%, rgba(12, 22, 38, 0.85) 50%, rgba(0, 242, 254, 0.10) 100%);
                border: 1px solid rgba(243, 112, 35, 0.35);
                border-radius: 18px;
                padding: 22px 28px;
                margin-bottom: 24px;
                backdrop-filter: blur(24px);
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.2);
                display: flex;
                align-items: center;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 16px;
            }

            .lpu-title-glow {
                font-size: 2.1rem;
                font-weight: 800;
                background: linear-gradient(135deg, #ffffff 20%, #ffe2d1 50%, #f37023 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                letter-spacing: -0.02em;
                margin: 0;
            }

            .lpu-sub-brand {
                font-size: 0.90rem;
                color: #8ed9ec;
                font-weight: 500;
                letter-spacing: 0.04em;
                margin-top: 4px;
            }

            /* AI Status Badges */
            .ai-pill {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                padding: 5px 14px;
                border-radius: 30px;
                font-size: 0.75rem;
                font-family: 'JetBrains Mono', monospace;
                font-weight: 600;
                letter-spacing: 0.05em;
                text-transform: uppercase;
            }

            .ai-pill-lpu {
                background: rgba(243, 112, 35, 0.15);
                color: #ff8533;
                border: 1px solid rgba(243, 112, 35, 0.5);
                box-shadow: 0 0 12px rgba(243, 112, 35, 0.25);
            }

            .ai-pill-cyan {
                background: rgba(0, 242, 254, 0.12);
                color: #00f2fe;
                border: 1px solid rgba(0, 242, 254, 0.45);
                box-shadow: 0 0 12px rgba(0, 242, 254, 0.25);
            }

            .ai-pill-green {
                background: rgba(56, 239, 125, 0.12);
                color: #38ef7d;
                border: 1px solid rgba(56, 239, 125, 0.45);
                box-shadow: 0 0 12px rgba(56, 239, 125, 0.25);
            }

            /* Pulsing live dot */
            .pulse-dot-lpu {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #f37023;
                box-shadow: 0 0 8px #f37023;
                animation: lpuPulse 1.8s infinite;
            }

            @keyframes lpuPulse {
                0% { box-shadow: 0 0 0 0 rgba(243, 112, 35, 0.7); }
                70% { box-shadow: 0 0 0 10px rgba(243, 112, 35, 0); }
                100% { box-shadow: 0 0 0 0 rgba(243, 112, 35, 0); }
            }

            /* Buttons: LPU Solar Ember Glow */
            .stButton > button {
                background: linear-gradient(135deg, rgba(243, 112, 35, 0.25) 0%, rgba(255, 138, 0, 0.35) 100%) !important;
                color: #ffffff !important;
                border: 1px solid rgba(243, 112, 35, 0.6) !important;
                border-radius: 12px !important;
                font-weight: 600 !important;
                padding: 10px 22px !important;
                box-shadow: 0 4px 18px rgba(243, 112, 35, 0.20) !important;
                transition: all 0.25s ease !important;
            }

            .stButton > button:hover {
                background: linear-gradient(135deg, rgba(243, 112, 35, 0.50) 0%, rgba(255, 138, 0, 0.60) 100%) !important;
                border-color: #ff8533 !important;
                box-shadow: 0 0 28px rgba(243, 112, 35, 0.45), 0 0 14px rgba(0, 242, 254, 0.3) !important;
                transform: translateY(-2px) !important;
            }

            /* Sidebar */
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #05080e 0%, #090e18 50%, #060a12 100%) !important;
                border-right: 1px solid rgba(243, 112, 35, 0.20) !important;
            }

            /* Inputs & Dropdowns */
            input, select, textarea {
                background: rgba(10, 18, 30, 0.85) !important;
                border: 1px solid rgba(243, 112, 35, 0.30) !important;
                color: #f0f6fc !important;
                border-radius: 10px !important;
            }
            input:focus, select:focus {
                border-color: #f37023 !important;
                box-shadow: 0 0 12px rgba(243, 112, 35, 0.35) !important;
            }
        </style>
    """, unsafe_allow_html=True)


def render_lpu_hero_header(faculty_name: str = None, subject_name: str = None):
    """Renders the top branding banner with the single round LPU seal and faculty context."""
    faculty_badge = f'<div class="ai-pill ai-pill-lpu">👨‍🏫 FACULTY: {faculty_name}</div>' if faculty_name else ''
    subj_badge = f'<div class="ai-pill ai-pill-cyan">📚 SUBJECT: {subject_name}</div>' if subject_name else ''
    b64 = get_round_logo_base64()
    logo_html = f'<img src="data:image/png;base64,{b64}" width="65" height="65" style="filter: drop-shadow(0 0 12px rgba(243, 112, 35, 0.45)); border-radius: 50%; flex-shrink: 0;" />' if b64 else ''

    st.markdown(f"""
        <div class="lpu-hero-banner">
            <div style="display: flex; align-items: center; gap: 18px;">
                {logo_html}
                <div>
                    <div style="font-size: 0.78rem; font-weight: 700; color: #ff8533; letter-spacing: 0.15em; text-transform: uppercase;">
                        Lovely Professional University • Punjab (India)
                    </div>
                    <h1 class="lpu-title-glow" style="margin: 2px 0;">Smart AI Face Recognition Attendance System</h1>
                    <div class="lpu-sub-brand">
                        Neural 128-D Biometric Engine • Dynamic Subject Timetable • Real-Time Recognition
                    </div>
                </div>
            </div>
            <div style="display: flex; gap: 8px; flex-wrap: wrap; align-items: center;">
                {faculty_badge}
                {subj_badge}
                <div class="ai-pill ai-pill-green">
                    ● VISION LIVE
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_lpu_kpi(title: str, value: str, subtitle: str = "", icon: str = "🎓", border_color: str = "#f37023"):
    """Renders an LPU Solar-Glacial KPI metric card."""
    st.markdown(f"""
        <div class="lpu-card" style="border-left: 4px solid {border_color};">
            <div class="lpu-kpi-title">
                <span>{icon}</span> {title}
            </div>
            <div class="lpu-kpi-val">
                {value}
            </div>
            {f'<div class="lpu-kpi-sub">{subtitle}</div>' if subtitle else ''}
        </div>
    """, unsafe_allow_html=True)


def render_lpu_status_strip(model_name: str = "YuNet DNN + SFace 128D", active_period: str = "Auto-Sync"):
    """Renders status pills across the top of panels."""
    st.markdown(f"""
        <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 18px; flex-wrap: wrap;">
            <div class="ai-pill ai-pill-lpu">
                <span class="pulse-dot-lpu"></span> LPU VISION ENGINE
            </div>
            <div class="ai-pill ai-pill-cyan">
                ⚡ {model_name}
            </div>
            <div class="ai-pill ai-pill-green">
                🛡️ LIVENESS DETECTOR ACTIVE
            </div>
            <div class="ai-pill ai-pill-lpu">
                📅 {active_period}
            </div>
        </div>
    """, unsafe_allow_html=True)
