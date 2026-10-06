import av
import cv2
import threading
import streamlit as st
import streamlit.components.v1 as components
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Deaf-Blind Safety AI",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ AI Environmental Safety Assistant")
st.caption("Live Safety Monitoring Prototype")


# =========================================================
# YOLO MODEL
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

    if area_ratio > 0.25:
        score += 25

    elif area_ratio > 0.12:
        score += 12

    elif area_ratio > 0.05:
        score += 5

    if approaching:
        score += 30

    return min(int(score), 100)


# =========================================================
# SAFETY LEVEL
# =========================================================

def get_level(score):

    if score >= 85:
        return "CRITICAL"

    elif score >= 65:
        return "HIGH"

    elif score >= 35:
        return "MEDIUM"

    return "LOW"


# =========================================================
# VIBRATION PATTERN
# =========================================================

def get_pattern(level):

    patterns = {
        "LOW": "• •",
        "MEDIUM": "● ● •",
        "HIGH": "● ● ● ●",
        "CRITICAL": "████████"
    }

    return patterns[level]


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

    if frame_count % 3 != 0 and last_annotated is not None:

        return av.VideoFrame.from_ndarray(
            last_annotated,
            format="bgr24"
        )

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

    # YOLO prediction
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

    # Detect objects
    if result.boxes is not None and len(result.boxes) > 0:

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for box, cls in zip(boxes, classes):

            x1, y1, x2, y2 = box.astype(int)

            class_id = int(cls)

            label = names[class_id]

            box_width = max(0, x2 - x1)
            box_height = max(0, y2 - y1)

            area = box_width * box_height

            area_ratio = area / frame_area

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

    # Safety level
    level = get_level(highest_score)

    pattern = get_pattern(level)

    # Update state
    with lock:

        state["label"] = highest_label
        state["score"] = highest_score
        state["level"] = level
        state["approaching"] = highest_approaching
        state["pattern"] = pattern

    # Draw YOLO boxes
    annotated = result.plot()

    # Alert message
    if level == "CRITICAL":

        message = "CRITICAL DANGER"

    elif level == "HIGH":

        message = "HIGH DANGER"

    elif level == "MEDIUM":

        message = "MEDIUM WARNING"

    else:

        message = "LOW RISK"

    # Alert panel
    cv2.rectangle(
        annotated,
        (8, 8),
        (620, 125),
        (0, 0, 0),
        -1
    )

    # Alert
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
# GPS NAVIGATION
# =========================================================

st.divider()

st.header("🗺️ Voice Walking Navigation")

st.write(
    "Enter your destination and start GPS walking navigation."
)

destination = st.text_input(
    "🎯 Enter destination",
    placeholder="Example: Ranipet Railway Station"
)


# =========================================================
# GPS HTML + JAVASCRIPT
# =========================================================

gps_html = """
<!DOCTYPE html>

<html>

<head>

<meta name="viewport" content="width=device-width, initial-scale=1.0">

<style>

body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 10px;
}

button {
    width: 100%;
    padding: 16px;
    font-size: 17px;
    font-weight: bold;
    border: none;
    border-radius: 12px;
    cursor: pointer;
}

#status {
    margin-top: 14px;
    padding: 12px;
    border-radius: 10px;
    font-size: 15px;
    line-height: 1.5;
}

</style>

</head>

<body>

<button onclick="startGPS()">
📍 START GPS VOICE NAVIGATION
</button>

<div id="status">
GPS navigation not started.
</div>


<script>

const destination = "__DESTINATION__";


function speak(message) {

    if ("speechSynthesis" in window) {

        window.speechSynthesis.cancel();

        const speech =
            new SpeechSynthesisUtterance(message);

        speech.rate = 0.9;
        speech.pitch = 1.0;
        speech.volume = 1.0;

        window.speechSynthesis.speak(speech);
    }
}


function setStatus(message) {

    document.getElementById("status").innerText = message;
}


function startGPS() {

    if (!destination.trim()) {

        setStatus(
            "Please enter a destination first."
        );

        speak(
            "Please enter a destination first."
        );

        return;
    }


    if (!navigator.geolocation) {

        setStatus(
            "GPS is not supported by this browser."
        );

        speak(
            "GPS is not supported by this browser."
        );

        return;
    }


    setStatus(
        "Getting your current GPS location..."
    );

    speak(
        "Getting your current GPS location."
    );


    navigator.geolocation.getCurrentPosition(

        function(position) {

            const latitude =
                position.coords.latitude;

            const longitude =
                position.coords.longitude;


            setStatus(
                "GPS location found. Opening walking navigation..."
            );


            speak(
                "Your location was found. Opening walking navigation."
            );


            const mapsURL =
                "https://www.google.com/maps/dir/?api=1" +
                "&origin=" +
                latitude +
                "," +
                longitude +
                "&destination=" +
                encodeURIComponent(destination) +
                "&travelmode=walking";


            setTimeout(

                function() {

                    window.open(
                        mapsURL,
                        "_blank"
                    );

                },

                1200
            );

        },


        function(error) {

            let message =
                "Unable to get your GPS location.";


            if (error.code === 1) {

                message =
                    "Location permission was denied. Please allow location access in your browser.";
            }


            else if (error.code === 2) {

                message =
                    "Your GPS location is unavailable. Please turn on location services.";
            }


            else if (error.code === 3) {

                message =
                    "GPS request timed out. Please try again.";
            }


            setStatus(message);

            speak(message);

        },


        {
            enableHighAccuracy: true,
            timeout: 15000,
            maximumAge: 0
        }

    );
}

</script>

</body>

</html>
"""


# =========================================================
# INSERT DESTINATION SAFELY
# =========================================================

safe_destination = (
    destination
    .replace("\\", "\\\\")
    .replace('"', '\\"')
    .replace("\n", " ")
    .replace("\r", " ")
)

gps_html = gps_html.replace(
    "__DESTINATION__",
    safe_destination
)


# Display GPS component
components.html(
    gps_html,
    height=230,
    scrolling=False
)


# =========================================================
# INFORMATION
# =========================================================

st.info(
    "📱 On your phone: enter a destination, "
    "tap START GPS VOICE NAVIGATION, "
    "allow location permission, and walking navigation "
    "will open in Google Maps."
)

st.caption(
    "The AI camera provides environmental obstacle awareness. "
    "Google Maps provides walking route and turn-by-turn navigation."
)
