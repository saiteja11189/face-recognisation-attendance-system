"""
Command-Line Interface (CLI) for Face Recognition Attendance System.
Supports period and subject-aware real-time monitoring, enrollment, reporting, and database management.
"""

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
import time
import cv2
import numpy as np

from core import (
    FaceDetector,
    FacePreprocessor,
    FaceEmbedder,
    SimilarityMatcher,
    Database,
    AttendanceManager,
    ScheduleManager,
    Camera,
    AntiSpoofingDetector,
    config,
)


def run_live_recognition(
    camera_index: int = config.DEFAULT_CAMERA_INDEX,
    metric: str = config.DEFAULT_METRIC,
    subject: str = None,
    period: str = None,
    enable_anti_spoofing: bool = True,
):
    """Launches desktop OpenCV live attendance tracking window with subject resolution."""
    db = Database()
    sched = ScheduleManager(db=db)

    # Resolve active subject if not manually specified
    now = datetime.now()
    active_slot = sched.get_current_period(dt=now)

    if not subject:
        if active_slot:
            subject = active_slot["subject_name"]
            period = active_slot["period_name"]
            period_start = active_slot["start_time"]
            print(f"[*] Auto-detected active timetable class: {period} - {subject} ({period_start})")
        else:
            subject = "General"
            period = "General"
            period_start = None
            print("[*] No active timetable period right now. Using default 'General' subject.")
    else:
        period = period or "General"
        period_start = None

    print("=" * 68)
    print("  FACE RECOGNITION ATTENDANCE SYSTEM - LIVE MONITOR")
    print("=" * 68)
    print(f"[*] Target Subject:      {subject} ({period})")
    print(f"[*] Camera Device Index: {camera_index}")
    print(f"[*] Similarity Metric:   {metric.upper()}")
    print(f"[*] Anti-Spoofing:       {'ENABLED' if enable_anti_spoofing else 'DISABLED'}")
    print("[*] Press 'q' or 'ESC' to Exit | Press 'r' to Reload Gallery")
    print("=" * 68)

    # Initialize modules
    detector = FaceDetector()
    embedder = FaceEmbedder()
    matcher = SimilarityMatcher(metric=metric)
    att_mgr = AttendanceManager(db=db)
    spoof_detector = AntiSpoofingDetector() if enable_anti_spoofing else None

    # Load enrolled gallery
    gallery = db.get_gallery(force_reload=True)
    print(f"[*] Loaded {len(gallery)} enrolled identities from database.")

    cam = Camera(source=camera_index)
    if not cam.start():
        print(f"[!] ERROR: Unable to open camera source {camera_index}.")
        return

    last_event_msg = f"System Ready. Marking for: {subject}"
    last_event_color = (255, 255, 255)
    last_event_time = time.time()

    cv2.namedWindow("Attendance Pro - Live Face Recognition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Attendance Pro - Live Face Recognition", 960, 720)

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]

            faces = detector.detect(frame)

            for face in faces:
                x, y, bw, bh = face.box
                face_crop = frame[y:y + bh, x:x + bw]

                # Check liveness
                is_live = True
                liveness_text = ""
                if spoof_detector and face_crop.size > 0:
                    liveness = spoof_detector.check_liveness(face_crop, face)
                    is_live = liveness.is_live
                    liveness_text = f"Live: {liveness.confidence:.0f}%" if is_live else "Spoof!"

                # Feature extraction & similarity matching
                embedding = embedder.process_face(frame, face)
                match_res = matcher.match(embedding, gallery)

                if not is_live:
                    box_color = (0, 0, 255)
                    label = f"SPOOF ALERT ({liveness_text})"
                elif match_res.is_match:
                    box_color = (0, 255, 0)
                    label = f"{match_res.name} ({match_res.confidence:.1f}%)"

                    # Log attendance for the specific subject/period
                    success, msg, _ = att_mgr.mark_attendance(
                        user_id=match_res.user_id,
                        confidence=match_res.confidence,
                        metric=metric,
                        subject=subject,
                        period=period,
                        period_start_time=period_start,
                    )
                    if success:
                        last_event_msg = f"[OK] {msg}"
                        last_event_color = (0, 255, 0)
                        last_event_time = time.time()
                    elif "Already marked" in msg:
                        last_event_msg = f"[INFO] {match_res.name}: Cooldown Active ({subject})"
                        last_event_color = (0, 255, 255)
                        last_event_time = time.time()
                else:
                    box_color = (0, 140, 255)
                    label = f"Unknown ({match_res.confidence:.1f}%)"

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), box_color, 2)

                # Draw name tag background
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x, y - th - 10), (x + tw + 10, y), box_color, -1)
                cv2.putText(
                    frame,
                    label,
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

                # Draw facial landmarks
                for lx, ly in face.landmarks:
                    cv2.circle(frame, (int(lx), int(ly)), 2, (0, 0, 255), -1)

            # Top HUD Bar
            cv2.rectangle(frame, (0, 0), (w, 42), (20, 20, 20), -1)
            hud_info = f"FPS: {cam.fps} | Class: {period} - {subject} | Enrolled: {len(gallery)}"
            cv2.putText(frame, hud_info, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2, cv2.LINE_AA)

            # Bottom Notification Banner
            if time.time() - last_event_time < 5.0:
                cv2.rectangle(frame, (0, h - 36), (w, h), (30, 30, 30), -1)
                cv2.putText(
                    frame,
                    last_event_msg,
                    (15, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    last_event_color,
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Attendance Pro - Live Face Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), 27]:
                break
            elif key == ord("r"):
                gallery = db.get_gallery(force_reload=True)
                last_event_msg = f"[INFO] Reloaded gallery: {len(gallery)} users."
                last_event_color = (255, 255, 0)
                last_event_time = time.time()

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("\n[*] Live monitoring session ended gracefully.")


def enroll_from_image(user_id: str, name: str, department: str, photo_path: str, email: str = ""):
    """Enrolls an individual using an existing high-resolution photo file."""
    path = Path(photo_path)
    if not path.exists():
        print(f"[!] ERROR: Photo file does not exist at {path}")
        return False

    img = cv2.imread(str(path))
    if img is None:
        print(f"[!] ERROR: Failed to read image from {path}")
        return False

    detector = FaceDetector()
    embedder = FaceEmbedder()
    db = Database()

    faces = detector.detect(img)
    if not faces:
        print("[!] ERROR: No face detected in the provided photo.")
        return False

    primary_face = faces[0]
    aligned = embedder.align_and_crop(img, primary_face)
    embedding = embedder.extract_embedding(aligned)

    dest_photo = config.FACES_DIR / f"{user_id}.jpg"
    cv2.imwrite(str(dest_photo), aligned)

    db.add_user(
        user_id=user_id,
        name=name,
        department=department,
        email=email,
        embedding=embedding,
        photo_path=str(dest_photo),
    )

    print(f"[+] SUCCESS: Enrolled {name} ({user_id}) from {path.name}")
    return True


def enroll_from_camera(user_id: str, name: str, department: str, email: str = "", num_samples: int = 5):
    """Enrolls an individual by capturing a burst of clean face frames from the webcam."""
    print("=" * 60)
    print(f"  ENROLLING: {name} (ID: {user_id}, Dept: {department})")
    print(f"  Collecting {num_samples} facial embedding samples...")
    print("  Look directly at the camera. Press 'c' to capture a sample, 'q' to cancel.")
    print("=" * 60)

    cam = Camera()
    if not cam.start():
        print("[!] ERROR: Could not open camera.")
        return False

    detector = FaceDetector()
    embedder = FaceEmbedder()
    db = Database()

    embeddings = []
    best_aligned_crop = None

    cv2.namedWindow("Enrollment Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Enrollment Camera", 800, 600)

    try:
        while len(embeddings) < num_samples:
            ret, frame = cam.read()
            if not ret or frame is None:
                continue

            display = frame.copy()
            faces = detector.detect(frame)

            msg = f"Samples: {len(embeddings)}/{num_samples} | Press 'c' to capture | 'q' to cancel"
            cv2.putText(display, msg, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if faces:
                primary = faces[0]
                x, y, w, h = primary.box
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display, f"Face OK: {primary.confidence * 100:.0f}%", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Enrollment Camera", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("[*] Enrollment cancelled by user.")
                return False
            elif key in [ord("c"), 32]:
                if not faces:
                    print("[!] No face detected. Please face the camera properly.")
                    continue
                aligned = embedder.align_and_crop(frame, faces[0])
                emb = embedder.extract_embedding(aligned)
                embeddings.append(emb)
                best_aligned_crop = aligned.copy()
                print(f"[+] Captured sample {len(embeddings)}/{num_samples}")

        composite_embedding = embedder.average_embeddings(embeddings)

        dest_photo = config.FACES_DIR / f"{user_id}.jpg"
        if best_aligned_crop is not None:
            cv2.imwrite(str(dest_photo), best_aligned_crop)

        db.add_user(
            user_id=user_id,
            name=name,
            department=department,
            email=email,
            embedding=composite_embedding,
            photo_path=str(dest_photo) if dest_photo.exists() else "",
        )
        print(f"\n[+] Successfully enrolled {name} ({user_id}) with composite averaged embedding!")
        return True

    finally:
        cam.stop()
        cv2.destroyAllWindows()


def print_report(date_str: str = None, subject: str = None, export_csv: bool = False, export_excel: bool = False):
    """Displays attendance report with subject breakdown."""
    db = Database()
    mgr = AttendanceManager(db=db)
    query_date = date_str or datetime.now().strftime("%Y-%m-%d")
    summary = mgr.get_daily_summary(date_str=query_date, subject=subject)

    print("\n" + "=" * 68)
    print(f"       ATTENDANCE REPORT FOR: {summary['date']} | SUBJECT: {summary['subject']}")
    print("=" * 68)
    print(f"Total Registered Users:  {summary['total_registered']}")
    print(f"Total Present Today:     {summary['total_present']}  ({summary['attendance_rate']}%)")
    print(f"Total Absent Today:      {summary['total_absent']}")
    print(f"  - On Time (Present):   {summary['present_count']}")
    print(f"  - Late Check-ins:      {summary['late_count']}")
    print("-" * 68)

    if summary["subject_breakdown"]:
        print("\nSubject Breakdown:")
        for sb, data in summary["subject_breakdown"].items():
            print(f"  * {sb:<25}: {data['count']} attended ({data['on_time']} on time, {data['late']} late)")

    if summary["records"]:
        print("\nAttendance Logs:")
        print(f"{'User ID':<10} {'Name':<18} {'Subject':<20} {'Period':<10} {'Time':<9} {'Status':<8}")
        print("-" * 80)
        for r in summary["records"][:30]:
            t_only = r["timestamp"].split()[-1]
            print(f"{r['user_id']:<10} {r['name']:<18} {r.get('subject', 'General'):<20} {r.get('period', 'General'):<10} {t_only:<9} {r['status']:<8}")

    if export_csv:
        csv_file = mgr.export_to_csv(query_date)
        print(f"\n[+] Exported CSV to: {csv_file}")
    if export_excel:
        xlsx_file = mgr.export_to_excel(query_date)
        print(f"[+] Exported Excel to: {xlsx_file}")
    print("=" * 68 + "\n")


def list_users():
    """Lists all registered identities."""
    db = Database()
    users = db.get_all_users()
    print("\n" + "=" * 70)
    print(f"   REGISTERED IDENTITIES IN DATABASE ({len(users)} Total)")
    print("=" * 70)
    print(f"{'User ID':<14} {'Name':<24} {'Department':<16} {'Registered On':<16}")
    print("-" * 70)
    for u in users:
        print(f"{u['user_id']:<14} {u['name']:<24} {u['department']:<16} {u['created_at']:<16}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Face Recognition Attendance System CLI (Subject & Period Aware)")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # live
    live_parser = subparsers.add_parser("live", help="Launch live attendance monitor")
    live_parser.add_argument("--camera", type=int, default=config.DEFAULT_CAMERA_INDEX, help="Webcam device index")
    live_parser.add_argument("--metric", choices=["cosine", "l2"], default=config.DEFAULT_METRIC, help="Similarity metric")
    live_parser.add_argument("--subject", default=None, help="Subject name (if omitted, auto-resolves from timetable)")
    live_parser.add_argument("--period", default=None, help="Period name (e.g. 'Period 1')")
    live_parser.add_argument("--no-spoof", action="store_true", help="Disable anti-spoofing filter")

    # enroll
    enroll_parser = subparsers.add_parser("enroll", help="Enroll a new identity")
    enroll_parser.add_argument("--id", required=True, help="User ID (Roll No / Employee ID)")
    enroll_parser.add_argument("--name", required=True, help="Full Name")
    enroll_parser.add_argument("--dept", default="General", help="Department / Class")
    enroll_parser.add_argument("--email", default="", help="Email address")
    enroll_parser.add_argument("--photo", default=None, help="Path to photo image (if omitted, uses webcam)")
    enroll_parser.add_argument("--samples", type=int, default=5, help="Number of webcam burst samples")

    # list-users
    subparsers.add_parser("list-users", help="List all enrolled users")

    # delete-user
    del_parser = subparsers.add_parser("delete-user", help="Delete a registered user")
    del_parser.add_argument("--id", required=True, help="User ID to delete")

    # report
    report_parser = subparsers.add_parser("report", help="View attendance report")
    report_parser.add_argument("--date", default=None, help="Date in YYYY-MM-DD format (default: today)")
    report_parser.add_argument("--subject", default=None, help="Filter by specific subject")
    report_parser.add_argument("--csv", action="store_true", help="Export to CSV file")
    report_parser.add_argument("--excel", action="store_true", help="Export to formatted Excel file")

    args = parser.parse_args()

    if args.command == "live":
        run_live_recognition(
            camera_index=args.camera,
            metric=args.metric,
            subject=args.subject,
            period=args.period,
            enable_anti_spoofing=not args.no_spoof,
        )
    elif args.command == "enroll":
        if args.photo:
            enroll_from_image(args.id, args.name, args.dept, args.photo, args.email)
        else:
            enroll_from_camera(args.id, args.name, args.dept, args.email, args.samples)
    elif args.command == "list-users":
        list_users()
    elif args.command == "delete-user":
        db = Database()
        if db.delete_user(args.id):
            print(f"[+] User {args.id} deleted successfully.")
        else:
            print(f"[!] User {args.id} not found in database.")
    elif args.command == "report":
        print_report(date_str=args.date, subject=args.subject, export_csv=args.csv, export_excel=args.excel)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
