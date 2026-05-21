# Import OpenCV for camera access, drawing boxes, and face detection
import cv2

# Import NumPy for handling image data as arrays of numbers
import numpy as np

# Import os for file/folder operations
import os

# Import pickle to save and load the trained model to disk
import pickle

# Import threading to run each camera in its own thread simultaneously
import threading

# Import time to track when alerts were last sent (cooldown)
import time

# Import datetime to timestamp detection logs
from datetime import datetime

# Import Path for clean file path handling
from pathlib import Path

# Import ctypes to show native Windows pop-up notifications
import ctypes

# Import winsound to play the Windows alert beep
import winsound

# ─── Configuration ────────────────────────────────────────────────────────────

# Folder containing one subfolder per known person
KNOWN_FACES_DIR  = "known_faces"

# Cached model file to avoid retraining every run
ENCODINGS_FILE   = "encodings.pkl"

# How strict face matching is — lower = stricter
CONFIDENCE_LIMIT = 70

# Haar cascade detection settings
SCALE         = 1.3
MIN_NEIGHBORS = 5

# Box colors in BGR
BOX_COLOR_KNOWN = (34, 158, 117)   # green
BOX_COLOR_UNK   = (56, 138, 221)   # blue
BOX_COLOR_ALERT = (0, 0, 220)      # red for watchlist hits

# Font for on-screen labels
FONT = cv2.FONT_HERSHEY_SIMPLEX

# How many seconds to wait before sending another alert for the same person
ALERT_COOLDOWN_SECONDS = 30

# File where detection events are logged
LOG_FILE = "detections.log"

# List of names to watch for — alerts fire when any of these are detected
# Edit this list to add or remove people you want to be notified about
# Empty list means alert on ANY known face
WATCHLIST = []

# How many camera indexes to try opening on startup (0, 1, 2, … up to this number)
MAX_CAMERAS_TO_TRY = 4

# ──────────────────────────────────────────────────────────────────────────────

# Load OpenCV's built-in Haar cascade face detector
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade  = cv2.CascadeClassifier(CASCADE_PATH)

# LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Lock so only one thread writes to the log file at a time
log_lock = threading.Lock()

# Tracks the last time an alert was sent for each person
last_alert_time: dict[str, float] = {}
alert_lock = threading.Lock()


# ── Notifications ─────────────────────────────────────────────────────────────

def windows_popup(title: str, message: str) -> None:
    """Show a native Windows message box pop-up."""
    # 0x40 = info icon, 0x1000 = always on top
    ctypes.windll.user32.MessageBoxW(0, message, title, 0x40 | 0x1000)


def send_alert(name: str, camera_id: int) -> None:
    """
    Fire a Windows notification when a watchlisted person is detected.
    Respects the cooldown so you don't get spammed every frame.
    """
    now = time.time()

    # Check cooldown — skip if we alerted for this person recently
    with alert_lock:
        last = last_alert_time.get(name, 0)
        if now - last < ALERT_COOLDOWN_SECONDS:
            return
        last_alert_time[name] = now

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"{name} detected on Camera {camera_id} at {timestamp}"

    # Write to log file
    write_log(msg)

    # Play the Windows default alert beep
    winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)

    # Show pop-up in a separate thread so it doesn't freeze the camera feed
    threading.Thread(target=windows_popup,
                     args=("Face Detected!", msg),
                     daemon=True).start()


def write_log(message: str) -> None:
    """Append a timestamped line to the detection log file."""
    with log_lock:
        with open(LOG_FILE, "a") as f:
            f.write(message + "\n")


def is_on_watchlist(name: str) -> bool:
    """Return True if this person should trigger an alert."""
    # Empty watchlist means alert on ANY recognized face
    if not WATCHLIST:
        return True
    return name in WATCHLIST


# ── Face recognition ──────────────────────────────────────────────────────────

def get_face_roi(gray: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Crop and resize a face to 100x100 for the LBPH recognizer."""
    return cv2.resize(gray[y:y+h, x:x+w], (100, 100))


def load_known_faces() -> tuple[list[str], bool]:
    """Load or train the LBPH face recognition model."""
    faces_dir = Path(KNOWN_FACES_DIR)

    # Create the folder if it doesn't exist yet
    if not faces_dir.exists():
        faces_dir.mkdir()
        print(f"[INFO] Created '{KNOWN_FACES_DIR}/' — add subfolders per person.")
        return [], False

    # Load from cache if available to skip retraining
    if os.path.exists(ENCODINGS_FILE):
        print("[INFO] Loading cached encodings…")
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
        recognizer.read(data["model_file"])
        return data["names"], True

    # Collect face images and labels from disk
    images, labels = [], []
    label_map: dict[str, int] = {}

    # Each subfolder in known_faces/ represents one person
    for person_dir in sorted(faces_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        name     = person_dir.name
        label_id = label_map.setdefault(name, len(label_map))

        # Load every image in this person's folder
        for img_path in person_dir.glob("*.[jp][pn]g"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, SCALE, MIN_NEIGHBORS)
            if len(faces) == 0:
                print(f"  ✗ {img_path.name} — no face, skipping")
                continue
            x, y, w, h = faces[0]
            images.append(get_face_roi(gray, x, y, w, h))
            labels.append(label_id)
            print(f"  ✓ {name} — {img_path.name}")

    if not images:
        return [], False

    # Build a name list indexed by label number
    id_to_name = {v: k for k, v in label_map.items()}
    name_list  = [id_to_name[i] for i in range(len(id_to_name))]

    # Train the LBPH recognizer on all collected face images
    recognizer.train(images, np.array(labels))
    model_file = "lbph_model.yml"
    recognizer.save(model_file)

    # Cache the model path and name list for next run
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"model_file": model_file, "names": name_list}, f)

    print(f"[INFO] Trained on {len(images)} image(s), {len(name_list)} person(s).")
    return name_list, True


# ── Per-camera thread ─────────────────────────────────────────────────────────

def camera_thread(camera_id: int, names: list[str], trained: bool,
                  stop_event: threading.Event) -> None:
    """
    Runs in its own thread — opens one camera, detects and recognizes
    faces, and fires alerts when a watchlisted person is seen.
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[WARN] Camera {camera_id} could not be opened — skipping.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(f"[INFO] Camera {camera_id} started.")

    window_name = f"Camera {camera_id} — Facial Recognition"

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print(f"[WARN] Camera {camera_id} lost — stopping thread.")
            break

        # Mirror the frame so it feels natural
        frame = cv2.flip(frame, 1)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Equalize histogram to handle dim or uneven lighting
        gray  = cv2.equalizeHist(gray)

        # Detect all faces in this frame
        faces = face_cascade.detectMultiScale(gray, SCALE, MIN_NEIGHBORS)

        for (x, y, w, h) in faces:
            label, conf, known = "Unknown", 0.0, False

            # Try to recognize the face if we have a trained model
            if trained:
                roi       = get_face_roi(gray, x, y, w, h)
                lid, dist = recognizer.predict(roi)
                if dist < CONFIDENCE_LIMIT and lid < len(names):
                    label = names[lid]
                    conf  = max(0.0, 1.0 - dist / 100)
                    known = True

            # Fire an alert if this person is on the watchlist
            if known and is_on_watchlist(label):
                threading.Thread(target=send_alert,
                                 args=(label, camera_id),
                                 daemon=True).start()

            # Choose box color: red for watchlist hit, green for known, blue for unknown
            if known and is_on_watchlist(label):
                color = BOX_COLOR_ALERT
            elif known:
                color = BOX_COLOR_KNOWN
            else:
                color = BOX_COLOR_UNK

            # Draw bounding box around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            # Draw name + confidence label above the box
            top_text = f"{label}  {conf:.0%}" if known else label
            (tw, th), _ = cv2.getTextSize(top_text, FONT, 0.55, 1)
            cv2.rectangle(frame, (x, y-th-10), (x+tw+8, y), color, -1)
            cv2.putText(frame, top_text, (x+4, y-6), FONT, 0.55, (255,255,255), 1)

        # HUD — info panel in the top-left corner
        watchlist_str = ", ".join(WATCHLIST) if WATCHLIST else "All known faces"
        hud = [
            f"Camera {camera_id}  |  Faces: {len(faces)}",
            f"Registered: {len(names)} people",
            f"Watchlist: {watchlist_str}",
            f"Cooldown: {ALERT_COOLDOWN_SECONDS}s  |  Log: {LOG_FILE}",
            "Q / ESC = quit all cameras",
        ]
        for i, line in enumerate(hud):
            cv2.putText(frame, line, (10, 25 + i*22), FONT, 0.50, (200,200,200), 1)

        cv2.imshow(window_name, frame)

        # Q or ESC on any camera window stops everything
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            stop_event.set()
            break

    cap.release()
    cv2.destroyWindow(window_name)
    print(f"[INFO] Camera {camera_id} closed.")


# ── Registration ──────────────────────────────────────────────────────────────

def register_faces_cli(names: list[str]) -> bool:
    """Open camera 0 to register a new face. Returns True if a face was saved."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera for registration.")
        return False

    print("\n[REGISTER] Look at camera. SPACE=capture  ESC=cancel")
    saved = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, "REGISTER — SPACE to capture, ESC to cancel",
                    (10, 30), FONT, 0.6, (0, 200, 255), 2)
        cv2.imshow("Register Face", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("[REGISTER] Cancelled.")
            break
        if key == 32:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, SCALE, MIN_NEIGHBORS)
            if len(faces) == 0:
                print("[REGISTER] No face detected — try again.")
                continue
            name = input("Enter name: ").strip()
            if not name:
                continue

            # Save the image into the person's subfolder
            save_dir = Path(KNOWN_FACES_DIR) / name
            save_dir.mkdir(parents=True, exist_ok=True)
            idx      = len(list(save_dir.glob("*.jpg")))
            img_path = save_dir / f"{idx:03d}.jpg"
            cv2.imwrite(str(img_path), frame)

            # Delete cache so it retrains with the new face next run
            for f in [ENCODINGS_FILE, "lbph_model.yml"]:
                if os.path.exists(f):
                    os.remove(f)

            print(f"[REGISTER] ✓ Saved '{name}' → {img_path}. Restart to retrain.")
            saved = True
            break

    cap.release()
    cv2.destroyWindow("Register Face")
    return saved


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    print("=" * 60)
    print("  Multi-Camera Facial Recognition + Alert System")
    print("=" * 60)

    # Startup menu
    print("\nOptions:")
    print("  1 — Start cameras (detection + alerts)")
    print("  2 — Register a new face first")
    print("  3 — Edit watchlist (currently:", WATCHLIST or "all known faces", ")")
    choice = input("\nChoice [1]: ").strip() or "1"

    if choice == "2":
        names, trained = load_known_faces()
        register_faces_cli(names)
        names, trained = load_known_faces()
    elif choice == "3":
        edit = input("Enter comma-separated names to watch (blank = all): ").strip()
        WATCHLIST.clear()
        if edit:
            WATCHLIST.extend([n.strip() for n in edit.split(",")])
        print(f"[INFO] Watchlist set to: {WATCHLIST or 'all known faces'}")

    # Load face recognition model
    names, trained = load_known_faces()

    # Auto-detect available cameras
    print(f"\n[INFO] Scanning for cameras (indexes 0–{MAX_CAMERAS_TO_TRY-1})…")
    available = []
    for idx in range(MAX_CAMERAS_TO_TRY):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            available.append(idx)
            cap.release()
    print(f"[INFO] Found {len(available)} camera(s): {available}")

    if not available:
        print("[ERROR] No cameras found. Plug in a camera and try again.")
        return

    # Shared stop event — when set, all camera threads quit
    stop_event = threading.Event()

    # Start one thread per camera
    threads = []
    for cam_id in available:
        t = threading.Thread(
            target=camera_thread,
            args=(cam_id, names, trained, stop_event),
            daemon=True
        )
        t.start()
        threads.append(t)

    print(f"\n[READY] {len(threads)} camera(s) running.")
    print(f"[INFO]  Watchlist: {WATCHLIST or 'all known faces'}")
    print(f"[INFO]  Alert cooldown: {ALERT_COOLDOWN_SECONDS}s")
    print(f"[INFO]  Detection log: {LOG_FILE}")
    print("[INFO]  Press Q or ESC in any camera window to quit all.\n")

    # Wait for all threads to finish
    for t in threads:
        t.join()

    print("\n[INFO] All cameras closed. Goodbye.")


if __name__ == "__main__":
    run()
