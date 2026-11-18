"""
Lovely Professional University (LPU) - Smart AI Face Recognition Attendance System.
Designed with Frosted Glacier Glassmorphism, Single Circular LPU Seal (No Words Beside),
Strict Persistent SMS OTP Verification (No Bypasses, Exact Match Only),
and Subject-Specific Faculty Isolation.
"""

from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Explicit modular imports
from core.detector import FaceDetector, DetectedFace
from core.preprocessor import FacePreprocessor
from core.embedder import FaceEmbedder
from core.matcher import SimilarityMatcher, MatchResult
from core.database import Database
from core.attendance_manager import AttendanceManager
from core.schedule_manager import ScheduleManager, PeriodSlot
from core.auth_manager import AuthManager
from core.anti_spoofing import AntiSpoofingDetector
import core.config as config
from core.visualizer import GlacialVisualizer
from ui.components import (
    apply_lpu_glacial_css,
    render_lpu_hero_header,
    render_lpu_kpi,
    render_lpu_status_strip,
    get_round_logo_base64,
)

ROUND_LOGO_PATH = Path("ui/assets/lpu_round_logo.png")

# Streamlit Page Setup
st.set_page_config(
    page_title="LPU - Smart AI Face Recognition Attendance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_lpu_glacial_css()


# Persistent Heavy AI Models Cached; Lightweight DB Managers Instantiated Freshly
@st.cache_resource
def load_ai_vision_models():
    """Caches heavy neural network weights once in memory."""
    detector = FaceDetector()
    embedder = FaceEmbedder()
    preprocessor = FacePreprocessor()
    spoof_detector = AntiSpoofingDetector()
    return detector, embedder, preprocessor, spoof_detector


detector, embedder, preprocessor, spoof_detector = load_ai_vision_models()

# Always initialize database & managers directly
db = Database()
att_mgr = AttendanceManager(db=db)
sched_mgr = ScheduleManager(db=db)
auth_mgr = AuthManager(db=db)

# Session State Initialization
if "authenticated_faculty" not in st.session_state:
    st.session_state.authenticated_faculty = None

if "metric" not in st.session_state:
    st.session_state.metric = config.DEFAULT_METRIC
if "cosine_threshold" not in st.session_state:
    st.session_state.cosine_threshold = config.COSINE_SIMILARITY_THRESHOLD
if "l2_threshold" not in st.session_state:
    st.session_state.l2_threshold = config.L2_DISTANCE_THRESHOLD
if "det_threshold" not in st.session_state:
    st.session_state.det_threshold = config.DETECTION_SCORE_THRESHOLD
if "cooldown_mins" not in st.session_state:
    st.session_state.cooldown_mins = config.ATTENDANCE_COOLDOWN_MINUTES
if "enable_antispoof" not in st.session_state:
    st.session_state.enable_antispoof = True

# Sync matcher with current session thresholds
matcher = SimilarityMatcher(
    metric=st.session_state.metric,
    cosine_threshold=st.session_state.cosine_threshold,
    l2_threshold=st.session_state.l2_threshold,
)
detector.set_score_threshold(st.session_state.det_threshold)
att_mgr.cooldown_minutes = st.session_state.cooldown_mins


# =============================================================
# VIEW 1: FACULTY / STAFF ID & PASSWORD AUTHENTICATION PORTAL
# =============================================================
if st.session_state.authenticated_faculty is None:
    col_l1, col_l2, col_l3 = st.columns([1, 2, 1])
    with col_l2:
        # Centered Single Round LPU Logo Seal (No Words Beside)
        b64_logo = get_round_logo_base64()
        if b64_logo:
            st.markdown(f"""
                <div style="text-align: center; margin-bottom: 16px;">
                    <img src="data:image/png;base64,{b64_logo}" width="135" style="filter: drop-shadow(0 0 20px rgba(243, 112, 35, 0.45)); border-radius: 50%;" />
                </div>
            """, unsafe_allow_html=True)

        st.markdown("""
            <div style="text-align: center; margin-bottom: 24px;">
                <h2 style="margin: 0; background: linear-gradient(135deg, #ffffff 10%, #ffd8be 60%, #f37023 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 1.85rem;">
                    FACULTY BIOMETRIC ACCESS PORTAL
                </h2>
                <div style="font-size: 0.85rem; color: #8ed9ec; margin-top: 6px;">
                    Staff ID & Password Authentication • Subject-Specific Class Attendance
                </div>
            </div>
        """, unsafe_allow_html=True)

        tab_login, tab_signup = st.tabs(["🔐 Sign In with Staff ID", "📝 Register New Faculty & Subject"])

        # TAB 1: SIGN IN WITH STAFF / USER ID & PASSWORD
        with tab_login:
            st.markdown("#### Enter your credentials:")
            login_ident = st.text_input("Staff / Faculty ID or Mobile Number:*", placeholder="e.g. EMP-001 or LPU-CSE-101", key="in_login_ident")
            login_pwd = st.text_input("Password:*", type="password", placeholder="Enter your password", key="in_login_pwd")

            if st.button("🔓 Sign In to Faculty Dashboard", type="primary", key="btn_login_submit"):
                if not login_ident.strip() or not login_pwd.strip():
                    st.error("Please enter both your Staff ID and password.")
                else:
                    ok, faculty_profile, msg = auth_mgr.login(login_ident, login_pwd)
                    if ok and faculty_profile:
                        st.session_state.authenticated_faculty = faculty_profile
                        st.success(f"✅ Welcome back, {faculty_profile['name']}!")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")

        # TAB 2: CREATE FACULTY ACCOUNT WITH USER-DEFINED SUBJECT & PASSWORD
        with tab_signup:
            st.markdown("#### Create your Faculty Profile & Credentials:")
            st.caption("Each faculty member registers with their Staff ID, password, and assigned subject.")

            new_name = st.text_input("Full Name:*", placeholder="e.g. Dr. Rajesh Kumar", key="reg_name").strip()
            new_fac_id = st.text_input("Staff / Faculty ID / Employee Code:*", placeholder="e.g. LPU-CSE-401", key="reg_fac_id").strip()
            new_phone = st.text_input("Mobile Number (10 Digits):*", placeholder="e.g. 9876543210", key="reg_phone_in").strip()
            new_dept = st.selectbox("Department:*", ["Computer Science & Engineering", "Artificial Intelligence & Data Science", "Information Technology", "Electronics & Comm", "Mechanical", "Management"], key="reg_dept")

            st.markdown("##### 📚 Your Teaching Subject:")
            new_subject = st.text_input("Subject Name:* (Enter your subject)", placeholder="e.g. Advanced Operating Systems, Cloud Architecture, etc.", key="reg_subj").strip()
            new_code = st.text_input("Subject Course Code:*", placeholder="e.g. CSE325", key="reg_code").strip()

            st.markdown("##### 🔑 Create Password:")
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                new_pwd = st.text_input("Create Password:*", type="password", placeholder="Min 4 characters", key="reg_pwd").strip()
            with p_col2:
                confirm_pwd = st.text_input("Confirm Password:*", type="password", placeholder="Re-enter password", key="reg_confirm_pwd").strip()

            if st.button("📝 Create Account & Register Subject", type="primary", key="btn_register_submit"):
                cleaned_reg = auth_mgr.clean_phone(new_phone)
                if not new_name or not new_fac_id or not new_subject or not new_pwd or not confirm_pwd:
                    st.error("Please fill in all mandatory fields (Name, Staff ID, Subject, and Password).")
                elif len(cleaned_reg) != 10:
                    st.error("Please enter a valid 10-digit mobile number.")
                elif new_pwd != confirm_pwd:
                    st.error("❌ Passwords do not match. Please retype carefully.")
                elif len(new_pwd) < 4:
                    st.error("❌ Password must be at least 4 characters long.")
                else:
                    r_ok, created_fac, r_msg = auth_mgr.register_faculty(
                        faculty_id=new_fac_id,
                        name=new_name,
                        phone_number=cleaned_reg,
                        department=new_dept,
                        subject_name=new_subject,
                        subject_code=new_code or "N/A",
                        password=new_pwd,
                    )
                    if r_ok and created_fac:
                        # Auto-add initial timetable slot for this faculty's subject
                        sched_mgr.add_slot(
                            day_of_week="Monday",
                            period_name="Period 1",
                            subject_code=new_code or "N/A",
                            subject_name=new_subject,
                            department=new_dept,
                            start_time="09:00",
                            end_time="10:00",
                            faculty_phone=cleaned_reg,
                        )
                        st.session_state.authenticated_faculty = created_fac
                        st.success(r_msg)
                        st.balloons()
                        st.rerun()
                    else:
                        st.error(f"❌ {r_msg}")

    st.stop()


# =============================================================
# VIEW 2: AUTHENTICATED FACULTY DASHBOARD
# =============================================================
faculty = st.session_state.authenticated_faculty
faculty_id = faculty.get("faculty_id", "EMP-001")
faculty_name = faculty["name"]
faculty_phone = faculty["phone_number"]
faculty_subject = faculty["subject_name"]
faculty_code = faculty.get("subject_code", "")
faculty_dept = faculty.get("department", "Computer Science & Engineering")

# SIDEBAR: SINGLE ROUND LPU SEAL & FACULTY DETAILS
b64_side_logo = get_round_logo_base64()
if b64_side_logo:
    st.sidebar.markdown(f"""
        <div style="text-align: center; margin-bottom: 14px;">
            <img src="data:image/png;base64,{b64_side_logo}" width="115" style="filter: drop-shadow(0 0 16px rgba(243, 112, 35, 0.40)); border-radius: 50%;" />
        </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown(f"""
    <div style="background: rgba(243, 112, 35, 0.12); border: 1px solid #f37023; border-radius: 12px; padding: 14px; margin-bottom: 15px;">
        <div style="font-size: 0.72rem; color: #ff8533; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">LOGGED IN FACULTY</div>
        <div style="font-size: 1.15rem; font-weight: 800; color: #ffffff; margin-top: 2px;">{faculty_name}</div>
        <div style="font-size: 0.78rem; color: #ff8533; font-family: monospace;">🆔 ID: {faculty_id}</div>
        <div style="font-size: 0.78rem; color: #8ed9ec; font-family: monospace;">📱 +91 {faculty_phone}</div>
        <div style="font-size: 0.80rem; color: #38ef7d; font-weight: 600; margin-top: 6px;">📚 Subject: {faculty_subject}</div>
        <div style="font-size: 0.74rem; color: #a5b4cb;">🏫 {faculty_dept}</div>
    </div>
""", unsafe_allow_html=True)

if st.sidebar.button("🚪 Sign Out (Switch Faculty)", type="secondary"):
    st.session_state.authenticated_faculty = None
    st.rerun()

menu = st.sidebar.radio(
    "Faculty Navigation",
    [
        "📸 AI Scanner & Attendance",
        "📅 My Subject Timetable",
        "📊 Subject Analytics & Shortage",
        "👤 Student Enrollment",
        "🗂️ Student Roster Gallery",
        "⚙️ AI Vision & Security Settings",
    ],
    index=0,
)

st.sidebar.markdown("---")
gallery = db.get_gallery()
st.sidebar.markdown(f"""
    <div style="background: rgba(0, 242, 254, 0.06); border: 1px solid rgba(0, 242, 254, 0.25); border-radius: 10px; padding: 10px; margin-bottom: 10px;">
        <div style="font-size: 0.72rem; color: #8ed9ec; text-transform: uppercase;">Enrolled Students</div>
        <div style="font-size: 1.3rem; font-weight: 700; color: #ffffff;">{len(gallery)} Identities</div>
    </div>
""", unsafe_allow_html=True)

subject_summary = att_mgr.get_daily_summary(subject=faculty_subject)
st.sidebar.markdown(f"""
    <div style="background: rgba(56, 239, 125, 0.06); border: 1px solid rgba(56, 239, 125, 0.25); border-radius: 10px; padding: 10px;">
        <div style="font-size: 0.72rem; color: #38ef7d; text-transform: uppercase;">{faculty_subject[:16]} Attendance</div>
        <div style="font-size: 1.3rem; font-weight: 700; color: #ffffff;">{subject_summary['attendance_rate']}%</div>
        <div style="font-size: 0.70rem; color: #ff8533;">{subject_summary['total_present']} Checked In Today</div>
    </div>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# TAB 1: AI SCANNER & ATTENDANCE (FACULTY SUBJECT SCOPED)
# -------------------------------------------------------------
if menu == "📸 AI Scanner & Attendance":
    render_lpu_hero_header(faculty_name=faculty_name, subject_name=faculty_subject)

    now = datetime.now()
    current_day = now.strftime("%A")
    current_time_str = now.strftime("%H:%M")

    # Resolve timetable slot specifically for this faculty's subject
    active_slot = sched_mgr.get_current_period(
        faculty_phone=faculty_phone,
        subject_name=faculty_subject,
        dt=now,
    )
    upcoming_slot = sched_mgr.get_upcoming_period(
        faculty_phone=faculty_phone,
        subject_name=faculty_subject,
        dt=now,
    )

    slot_label = f"{active_slot['period_name']}: {active_slot['subject_name']}" if active_slot else f"Manual Period: {faculty_subject}"
    render_lpu_status_strip(model_name="YuNet + SFace 128D", active_period=f"{current_day.upper()} | {slot_label}")

    col_sched1, col_sched2 = st.columns([2, 1])

    with col_sched1:
        if active_slot:
            st.markdown(f"""
                <div style="background: rgba(243, 112, 35, 0.12); border: 1px solid #f37023; border-radius: 12px; padding: 14px 18px; margin-bottom: 15px;">
                    <span style="color: #ff8533; font-weight: 700; font-size: 0.85rem; letter-spacing: 0.08em;">● ONGOING CLASS FOR {faculty_name.upper()}</span>
                    <h3 style="margin: 2px 0; color: #ffffff;">{active_slot['period_name']}: {active_slot['subject_name']}</h3>
                    <span style="color: #bbf0ff; font-size: 0.85rem;">Class Time: {active_slot['start_time']} - {active_slot['end_time']} | Dept: {active_slot['department']}</span>
                </div>
            """, unsafe_allow_html=True)
            target_period = active_slot["period_name"]
            period_start = active_slot["start_time"]
        else:
            up_text = f"Next scheduled class: {upcoming_slot['period_name']} ({upcoming_slot['start_time']})" if upcoming_slot else "No more timetable slots today"
            st.markdown(f"""
                <div style="background: rgba(0, 242, 254, 0.08); border: 1px solid rgba(0, 242, 254, 0.35); border-radius: 12px; padding: 14px 18px; margin-bottom: 15px;">
                    <span style="color: #00f2fe; font-weight: 700; font-size: 0.85rem; letter-spacing: 0.08em;">ℹ️ OUTSIDE SCHEDULED TIMETABLE HOURS</span>
                    <h4 style="margin: 2px 0; color: #ffffff;">{current_day}, {current_time_str}</h4>
                    <span style="color: #bbf0ff; font-size: 0.85rem;">{up_text}</span>
                </div>
            """, unsafe_allow_html=True)
            target_period = "Period 1"
            period_start = None

    with col_sched2:
        target_period = st.selectbox(
            "Select Period for Attendance:",
            ["Period 1", "Period 2", "Period 3", "Period 4", "Period 5", "Lab Session", "Extra Lecture"],
            index=0,
        )

    st.markdown(f"🎯 **Subject Locked to:** `<span style='color: #f37023; font-weight: 700; font-size: 1.1rem;'>{faculty_subject}</span>` ({target_period})", unsafe_allow_html=True)
    st.markdown("---")

    cam_mode = st.radio(
        "Biometric Sensor Source:",
        ["🌐 Browser Web Camera (Instant Snap)", "📁 Upload Classroom Photo"],
        horizontal=True,
    )

    input_image = None

    if cam_mode == "🌐 Browser Web Camera (Instant Snap)":
        cam_snap = st.camera_input("Capture live face frame to take attendance:")
        if cam_snap is not None:
            pil_snap = Image.open(cam_snap)
            input_image = cv2.cvtColor(np.array(pil_snap), cv2.COLOR_RGB2BGR)
    else:
        up_file = st.file_uploader("Upload Classroom Photo (JPG, PNG):", type=["jpg", "jpeg", "png"])
        if up_file is not None:
            pil_up = Image.open(up_file)
            input_image = cv2.cvtColor(np.array(pil_up), cv2.COLOR_RGB2BGR)

    if input_image is not None:
        with st.spinner(f"Extracting 128-D Embeddings and logging for {faculty_subject}..."):
            faces = detector.detect(input_image)

            st.markdown(f"### ⚡ Biometric Vision: <span style='color: #f37023;'>{len(faces)} Target(s) Acquired</span>", unsafe_allow_html=True)

            if not faces:
                st.warning("No face detected by YuNet. Face camera under clear lighting.")
            else:
                annotated_img = input_image.copy()
                match_results = []

                cols_display = st.columns([5, 3])

                for idx, face in enumerate(faces):
                    x, y, bw, bh = face.box
                    face_crop = input_image[y:y + bh, x:x + bw]

                    # Anti-spoofing check
                    is_live = True
                    liveness_reason = "Live Human"
                    if st.session_state.enable_antispoof and face_crop.size > 0:
                        liveness = spoof_detector.check_liveness(face_crop, face)
                        is_live = liveness.is_live
                        liveness_reason = liveness.reason

                    # 128D Embedding extraction & similarity match
                    embedding = embedder.process_face(input_image, face)
                    match_res = matcher.match(embedding, gallery)

                    # Mark attendance scoped to this specific faculty's subject & period
                    if not is_live:
                        status_str = f"BLOCKED: {liveness_reason}"
                        status_color = "#ff4444"
                    elif match_res.is_match:
                        success, log_msg, rec = att_mgr.mark_attendance(
                            user_id=match_res.user_id,
                            confidence=match_res.confidence,
                            metric=st.session_state.metric,
                            subject=faculty_subject,
                            period=target_period,
                            period_start_time=period_start,
                        )
                        status_str = log_msg
                        status_color = "#38ef7d" if success else "#ffbb33"
                    else:
                        status_str = "Unknown Identity - Unregistered in LPU Roster"
                        status_color = "#f37023"

                    # Render LPU Glacial HUD overlay
                    annotated_img = GlacialVisualizer.render_hud_overlay(
                        image=annotated_img,
                        face=face,
                        match=match_res,
                        is_live=is_live,
                        subject=faculty_subject,
                        period=target_period,
                    )

                    match_results.append({
                        "face_idx": idx + 1,
                        "name": match_res.name,
                        "user_id": match_res.user_id or "UNREGISTERED",
                        "confidence": match_res.confidence,
                        "score": match_res.score,
                        "status_str": status_str,
                        "status_color": status_color,
                        "is_live": is_live,
                    })

                with cols_display[0]:
                    rgb_hud = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                    st.image(rgb_hud, caption=f"LPU HUD Output ({faculty_subject})", use_container_width=True)

                with cols_display[1]:
                    st.markdown(f"#### 🔬 Verified Attendees for {faculty_subject}")
                    for m in match_results:
                        st.markdown(f"""
                            <div class="lpu-card" style="margin-bottom: 12px; border-left: 4px solid {m['status_color']};">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <strong style="font-size: 1.05rem; color: #ffffff;">{m['name']}</strong>
                                    <span style="font-family: monospace; color: #00f2fe; font-size: 0.85rem;">{m['confidence']:.1f}% CONF</span>
                                </div>
                                <div style="font-size: 0.8rem; color: #8ed9ec; margin-top: 4px;">
                                    Roll No: <code>{m['user_id']}</code> | Subject: <code>{faculty_subject}</code>
                                </div>
                                <div style="font-size: 0.82rem; color: {m['status_color']}; font-weight: 600; margin-top: 6px;">
                                    {m['status_str']}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)


# -------------------------------------------------------------
# TAB 2: MY SUBJECT TIMETABLE (FACULTY-DEFINED SLOTS)
# -------------------------------------------------------------
elif menu == "📅 My Subject Timetable":
    render_lpu_hero_header(faculty_name=faculty_name, subject_name=faculty_subject)
    st.markdown(f"### 📅 Weekly Schedule for **{faculty_subject}** ({faculty_name})")

    tab_my_sched, tab_add_slot = st.tabs(["📋 View My Timetable Slots", "➕ Add Class Period Slot"])

    with tab_my_sched:
        my_slots = sched_mgr.get_all_slots(
            faculty_phone=faculty_phone,
            subject_name=faculty_subject,
        )

        if my_slots:
            df_my = pd.DataFrame(my_slots)[["id", "day_of_week", "period_name", "subject_name", "department", "start_time", "end_time"]]
            df_my.columns = ["Slot ID", "Day of Week", "Period", "Subject", "Department", "Start Time", "End Time"]
            st.dataframe(df_my, use_container_width=True)

            del_c1, _ = st.columns([1, 3])
            with del_c1:
                del_slot_id = st.selectbox("Delete Period Slot:", [s["id"] for s in my_slots])
                if st.button("Delete Slot", type="secondary"):
                    sched_mgr.delete_slot(del_slot_id)
                    st.warning(f"Deleted Slot #{del_slot_id}")
                    st.rerun()
        else:
            st.info(f"No timetable slots added yet for {faculty_subject}. Add your weekly periods in the tab below.")

    with tab_add_slot:
        st.subheader(f"Add a new class timing for: {faculty_subject}")
        with st.form("add_my_slot_form"):
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                slot_day = st.selectbox("Day of Week:", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"])
                slot_period_name = st.selectbox("Period Name:", ["Period 1", "Period 2", "Period 3", "Period 4", "Period 5", "Lab Session", "Tutorial"])
            with f_col2:
                tc1, tc2 = st.columns(2)
                with tc1:
                    slot_start = st.text_input("Start Time (HH:MM):", value="09:00")
                with tc2:
                    slot_end = st.text_input("End Time (HH:MM):", value="10:00")

            if st.form_submit_button(f"Add Slot to {faculty_subject}", type="primary"):
                sched_mgr.add_slot(
                    day_of_week=slot_day,
                    period_name=slot_period_name,
                    subject_code=faculty_code or "N/A",
                    subject_name=faculty_subject,
                    department=faculty_dept,
                    start_time=slot_start,
                    end_time=slot_end,
                    faculty_phone=faculty_phone,
                )
                st.success(f"Added {slot_period_name} ({slot_start} - {slot_end}) on {slot_day} for {faculty_subject}!")
                st.rerun()


# -------------------------------------------------------------
# TAB 3: SUBJECT ATTENDANCE ANALYTICS & 75% SHORTAGE
# -------------------------------------------------------------
elif menu == "📊 Subject Analytics & Shortage":
    render_lpu_hero_header(faculty_name=faculty_name, subject_name=faculty_subject)
    st.markdown(f"### 📊 Attendance Records & Shortage Tracker for **{faculty_subject}**")

    q_date = st.date_input("Filter by Date:", value=datetime.now()).strftime("%Y-%m-%d")

    summary = att_mgr.get_daily_summary(date_str=q_date, subject=faculty_subject)

    # Glacial KPI Cards for this faculty's subject
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_lpu_kpi("Enrolled Students", str(summary["total_registered"]), "In active roster", icon="👥", border_color="#f37023")
    with k2:
        render_lpu_kpi(f"Present ({faculty_subject[:10]})", str(summary["total_present"]), f"{summary['attendance_rate']}% of roster", icon="✅", border_color="#38ef7d")
    with k3:
        render_lpu_kpi("Absent Students", str(summary["total_absent"]), "Requires follow-up", icon="❌", border_color="#ff5555")
    with k4:
        render_lpu_kpi("On Time Rate", f"{summary['present_count']}", f"{summary['late_count']} Late check-ins", icon="⏱️", border_color="#ffaa00")

    st.markdown("---")

    t_logs, t_shortage = st.tabs([f"📋 {faculty_subject} Logs ({q_date})", f"⚠️ 75% Attendance Shortage Tracker ({faculty_subject})"])

    with t_logs:
        exp1, exp2, _ = st.columns([1, 1, 3])
        with exp1:
            csv_f = att_mgr.export_to_csv(q_date)
            with open(csv_f, "rb") as f:
                st.download_button("📥 Download CSV Report", f.read(), f"attendance_{faculty_subject}_{q_date}.csv", "text/csv")
        with exp2:
            xlsx_f = att_mgr.export_to_excel(q_date)
            with open(xlsx_f, "rb") as f:
                st.download_button("📊 Download Excel Report", f.read(), f"attendance_{faculty_subject}_{q_date}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        if summary["records"]:
            df = pd.DataFrame(summary["records"])[["user_id", "name", "subject", "period", "timestamp", "status", "confidence"]]
            df.columns = ["Roll No / ID", "Name", "Subject", "Period", "Timestamp", "Status", "Confidence (%)"]
            st.dataframe(df, use_container_width=True)
        else:
            st.info(f"No records logged for {faculty_subject} on {q_date}.")

    with t_shortage:
        st.subheader(f"⚠️ Mandatory 75% Attendance Shortage Monitor for: {faculty_subject}")
        st.caption(f"Students with overall attendance below 75% in {faculty_subject} are flagged for debarment warning.")

        student_analytics = att_mgr.get_student_subject_analytics()

        if student_analytics:
            shortage_rows = []
            for s in student_analytics:
                subj_data = s["subject_breakdown"].get(faculty_subject, {"attended": 0, "conducted": 0, "percentage": 0.0})
                rate = subj_data["percentage"]
                is_short = rate < 75.0

                shortage_rows.append({
                    "Roll No / ID": s["user_id"],
                    "Student Name": s["name"],
                    "Department": s["department"],
                    "Sessions Attended": f"{subj_data['attended']} / {subj_data['conducted']}",
                    f"{faculty_subject} Attendance Rate": f"{rate}%",
                    "Eligibility Status": "⚠️ ATTENDANCE SHORTAGE (< 75%)" if is_short else "✅ ELIGIBLE (>= 75%)",
                })
            df_sh = pd.DataFrame(shortage_rows)
            st.dataframe(df_sh, use_container_width=True)


# -------------------------------------------------------------
# TAB 4: STUDENT ENROLLMENT
# -------------------------------------------------------------
elif menu == "👤 Student Enrollment":
    render_lpu_hero_header(faculty_name=faculty_name, subject_name=faculty_subject)
    st.markdown("### 👤 LPU Student Biometric Identity Enrollment")

    col_f, col_c = st.columns(2)
    with col_f:
        st.subheader("Student Credentials")
        r_id = st.text_input("Registration / Roll No (e.g. 12204567)*").strip()
        r_name = st.text_input("Full Name*", placeholder="e.g. Aryan Sharma").strip()
        r_dept = st.selectbox("Department / School*", ["Computer Science & Engineering", "Artificial Intelligence & Data Science", "Information Technology", "General"])
        r_email = st.text_input("LPU Email Address:", placeholder="aryan.12204567@lpu.in").strip()
        src_mode = st.radio("Capture Lens:", ["Webcam Snap", "Photo Upload"], horizontal=True)

    with col_c:
        st.subheader("Biometric Optical Acquisition")
        reg_img = None
        if src_mode == "Webcam Snap":
            s_snap = st.camera_input("Capture Portrait:")
            if s_snap is not None:
                reg_img = cv2.cvtColor(np.array(Image.open(s_snap)), cv2.COLOR_RGB2BGR)
        else:
            up_f = st.file_uploader("Upload Portrait (JPG, PNG):", type=["jpg", "png", "jpeg"])
            if up_f is not None:
                reg_img = cv2.cvtColor(np.array(Image.open(up_f)), cv2.COLOR_RGB2BGR)

    if reg_img is not None and r_id and r_name:
        st.markdown("---")
        detected = detector.detect(reg_img)
        if not detected:
            st.error("No face detected by YuNet. Face camera under clear lighting.")
        else:
            primary = detected[0]
            aligned = embedder.align_and_crop(reg_img, primary)

            prev_c1, prev_c2 = st.columns([1, 3])
            with prev_c1:
                st.image(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB), caption="112x112 Aligned Geometry", width=140)
            with prev_c2:
                st.write(f"**Target Name:** {r_name} (`{r_id}`)")
                st.write(f"**Department:** {r_dept}")
                st.write(f"**YuNet Score:** {primary.confidence * 100:.1f}%")

                if st.button("Confirm Enrollment & Generate 128-D Latent Vector", type="primary"):
                    embedding = embedder.extract_embedding(aligned)
                    photo_dest = config.FACES_DIR / f"{r_id}.jpg"
                    cv2.imwrite(str(photo_dest), aligned)

                    db.add_user(r_id, r_name, r_dept, embedding, r_email, str(photo_dest))
                    st.success(f"Successfully registered {r_name} into LPU Biometric Vault!")
                    st.balloons()


# -------------------------------------------------------------
# TAB 5: STUDENT ROSTER GALLERY
# -------------------------------------------------------------
elif menu == "🗂️ Student Roster Gallery":
    render_lpu_hero_header(faculty_name=faculty_name, subject_name=faculty_subject)
    st.markdown("### 🗂️ LPU Enrolled Students Gallery")

    users = db.get_all_users()
    st.markdown(f"**Total Registered Students:** `<span style='color: #f37023; font-weight: 700;'>{len(users)}</span>`", unsafe_allow_html=True)

    if not users:
        st.info("No students enrolled yet. Head to 'Student Enrollment' to add students.")
    else:
        g_cols = st.columns(4)
        for idx, u in enumerate(users):
            with g_cols[idx % 4]:
                st.markdown(f"""
                    <div class="lpu-card" style="text-align: center; margin-bottom: 16px;">
                        <div style="font-weight: 700; color: #ffffff; font-size: 1.05rem; margin-top: 8px;">{u['name']}</div>
                        <div style="font-size: 0.8rem; color: #ff8533; font-family: monospace;">ROLL: {u['user_id']}</div>
                        <div style="font-size: 0.78rem; color: #8ed9ec;">{u['department']}</div>
                        <div style="font-size: 0.72rem; color: #64b5f6; margin-top: 4px;">Registered: {u['created_at'][:10]}</div>
                    </div>
                """, unsafe_allow_html=True)
                p_path = Path(u.get("photo_path", ""))
                if p_path.exists():
                    st.image(str(p_path), width=120)
                if st.button("Delete Student", key=f"del_u_{u['user_id']}", type="secondary"):
                    db.delete_user(u["user_id"])
                    st.warning(f"Deleted {u['name']}")
                    st.rerun()


# -------------------------------------------------------------
# TAB 6: AI VISION & SECURITY SETTINGS
# -------------------------------------------------------------
elif menu == "⚙️ AI Vision & Security Settings":
    render_lpu_hero_header(faculty_name=faculty_name, subject_name=faculty_subject)
    st.markdown("### ⚙️ Deep Learning Latent Space & Faculty Account Security")

    c_cal1, c_cal2 = st.columns(2)
    with c_cal1:
        st.subheader("Deep Learning Latent Space")
        s_metric = st.selectbox("Distance Metric:", ["cosine", "l2"], index=0 if st.session_state.metric == "cosine" else 1)
        st.session_state.metric = s_metric

        if s_metric == "cosine":
            c_th = st.slider("Cosine Similarity Threshold:", 0.20, 0.70, float(st.session_state.cosine_threshold), 0.01)
            st.session_state.cosine_threshold = c_th
        else:
            l_th = st.slider("Euclidean (L2) Distance Threshold:", 0.50, 1.50, float(st.session_state.l2_threshold), 0.01)
            st.session_state.l2_threshold = l_th

        d_th = st.slider("YuNet Detection Confidence Threshold:", 0.50, 0.95, float(st.session_state.det_threshold), 0.05)
        st.session_state.det_threshold = d_th

        cooldown = st.number_input("Subject Cooldown Window (Minutes):", 1, 720, int(st.session_state.cooldown_mins))
        st.session_state.cooldown_mins = cooldown

        st.markdown("---")
        spoof_on = st.checkbox("Passive Anti-Spoofing Filter", value=st.session_state.enable_antispoof)
        st.session_state.enable_antispoof = spoof_on

    with c_cal2:
        st.subheader("🔐 Change Account Password")
        st.caption(f"Update login credentials for Staff ID: **{faculty_id}**")

        old_pass = st.text_input("Current Password:*", type="password", key="cp_old_pass")
        new_pass1 = st.text_input("New Password:*", type="password", key="cp_new_pass1")
        new_pass2 = st.text_input("Confirm New Password:*", type="password", key="cp_new_pass2")

        if st.button("🔑 Update Password", type="primary", key="btn_update_pass"):
            if not old_pass or not new_pass1 or not new_pass2:
                st.error("Please fill in all password fields.")
            elif new_pass1 != new_pass2:
                st.error("New passwords do not match.")
            elif len(new_pass1) < 4:
                st.error("New password must be at least 4 characters long.")
            else:
                c_ok, c_msg = auth_mgr.change_password(faculty_id, old_pass.strip(), new_pass1.strip())
                if c_ok:
                    st.success(f"✅ {c_msg}")
                else:
                    st.error(f"❌ {c_msg}")

        st.markdown("---")
        st.subheader("🛠️ System Reset")
        if st.button("Reset AI Hyperparameters to Defaults"):
            st.session_state.metric = config.DEFAULT_METRIC
            st.session_state.cosine_threshold = config.COSINE_SIMILARITY_THRESHOLD
            st.session_state.l2_threshold = config.L2_DISTANCE_THRESHOLD
            st.session_state.det_threshold = config.DETECTION_SCORE_THRESHOLD
            st.session_state.cooldown_mins = config.ATTENDANCE_COOLDOWN_MINUTES
            st.session_state.enable_antispoof = True
            st.success("Factory defaults restored!")
            st.rerun()
