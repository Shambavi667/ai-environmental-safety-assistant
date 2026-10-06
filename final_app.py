import av
import cv2
import threading
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer

st.set_page_config(
    page_title="Deaf-Blind Safety AI",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ AI Environmental Safety Assistant")
st.caption("Live Safety Monitoring Prototype")


# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_model():
    return YOLO("yolo26n.pt")


model = load_model()


# =========================================================
# SHARED STATE
# =========================================================

lock = threading.Lock()

state = {
    "label": "None",
    "score": 0,
    "level": "LOW",
    "approaching": False,
    "pattern": "• •"
}

previous_areas = {}


# =========================================================
# RISK CALCULATION
# =========================================================

def calculate_risk(label, area_ratio, approaching):

    base = {
        "car": 55,
        "truck": 60,
        "bus": 60,
        "motorcycle": 55,
        "bicycle": 30,
        "person": 15,
        "chair": 5,
        "bed": 5,
        "laptop": 5,
        "clock": 2,
        "bottle": 3,
    }

    score = base.get(label, 5)

    # Larger object = approximate closeness
    if area_ratio > 0.25:
        score += 25
    elif area_ratio > 0.12:
        score += 12
    elif area_ratio > 0.05:
        score += 5

    # Additional risk if approaching detection is enabled
    if approaching:
        score += 30

    return min(int(score), 100)


def get_level(score):

    if score >= 85:
        return "CRITICAL"
    elif score >= 65:
        return "HIGH"
    elif score >= 35:
        return "MEDIUM"

    return "LOW"


def get_pattern(level):

    return {
        "LOW": "• •",
        "MEDIUM": "● ● •",
        "HIGH": "● ● ● ●",
        "CRITICAL": "████████"
    }[level]


# =========================================================
# VIDEO PROCESSING
# =========================================================

frame_count = 0
last_annotated = None


def video_frame_callback(frame):

    global frame_count
    global last_annotated

    img = frame.to_ndarray(format="bgr24")

    frame_count += 1

    # Process every 3rd frame for smoother video
    if frame_count % 3 != 0 and last_annotated is not None:

        return av.VideoFrame.from_ndarray(
            last_annotated,
            format="bgr24"
        )

    # -----------------------------------------------------
    # Resize frame for faster processing
    # -----------------------------------------------------

    original_h, original_w = img.shape[:2]

    max_width = 640

    if original_w > max_width:

        scale = max_width / original_w

        new_w = int(original_w * scale)
        new_h = int(original_h * scale)

        small = cv2.resize(
            img,
            (new_w, new_h)
        )

    else:
        small = img.copy()

    h, w = small.shape[:2]

    frame_area = h * w

    # -----------------------------------------------------
    # YOLO DETECTION
    # -----------------------------------------------------

    results = model.predict(
        small,
        conf=0.45,
        imgsz=416,
        verbose=False
    )

    result = results[0]

    names = result.names

    highest_score = 0
    highest_label = "None"
    highest_approaching = False

    # -----------------------------------------------------
    # PROCESS DETECTIONS
    # -----------------------------------------------------

    if result.boxes is not None and len(result.boxes) > 0:

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for box, cls in zip(boxes, classes):

            x1, y1, x2, y2 = box.astype(int)

            class_id = int(cls)

            label = names[class_id]

            width = max(0, x2 - x1)
            height = max(0, y2 - y1)

            area = width * height

            area_ratio = area / frame_area

            # Approach detection can be added later
            approaching = False

            score = calculate_risk(
                label,
                area_ratio,
                approaching
            )

            if score > highest_score:

                highest_score = score
                highest_label = label
                highest_approaching = approaching

    # -----------------------------------------------------
    # RISK LEVEL
    # -----------------------------------------------------

    level = get_level(highest_score)

    pattern = get_pattern(level)

    # -----------------------------------------------------
    # UPDATE SHARED STATE
    # -----------------------------------------------------

    with lock:

        state["label"] = highest_label
        state["score"] = highest_score
        state["level"] = level
        state["approaching"] = highest_approaching
        state["pattern"] = pattern

    # -----------------------------------------------------
    # DRAW YOLO DETECTIONS
    # -----------------------------------------------------

    annotated = result.plot()

    # -----------------------------------------------------
    # DISPLAY WARNING
    # -----------------------------------------------------

    if level == "CRITICAL":

        message = "CRITICAL DANGER"

    elif level == "HIGH":

        message = "HIGH DANGER"

    elif level == "MEDIUM":

        message = "MEDIUM WARNING"

    else:

        message = "LOW RISK"

    # Information box
    cv2.rectangle(
        annotated,
        (8, 8),
        (620, 125),
        (0, 0, 0),
        -1
    )

    # Risk
    cv2.putText(
        annotated,
        message,
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    # Object
    cv2.putText(
        annotated,
        f"Object: {highest_label}",
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )

    # Danger score
    cv2.putText(
        annotated,
        f"Danger Score: {highest_score}/100",
        (20, 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )

    # Vibration pattern
    cv2.putText(
        annotated,
        f"Vibration: {pattern}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1
    )

    last_annotated = annotated

    return av.VideoFrame.from_ndarray(
        annotated,
        format="bgr24"
    )


# =========================================================
# LIVE CAMERA
# =========================================================

st.subheader("📷 Live Environmental Monitoring")

webrtc_streamer(
    key="safety-camera",

    video_frame_callback=video_frame_callback,

    media_stream_constraints={
        "video": True,
        "audio": False
    }
)


# =========================================================
# NAVIGATION
# =========================================================

st.divider()

st.header("🗺️ Voice Walking Navigation")

st.write(
    "Use the AI camera for obstacle awareness. "
    "GPS navigation will be connected separately."
)

st.info(
    "GPS navigation module will be added after "
    "the main Streamlit application is deployed."
)
