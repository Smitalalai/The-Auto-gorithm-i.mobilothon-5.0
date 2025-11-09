import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib


df = pd.read_csv("/content/synthetic_metrics.csv")


X = df.drop(columns=["timestamp", "label"])


le = LabelEncoder()
y = le.fit_transform(df["label"])


print("\nLabel Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("\n Model Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))


joblib.dump(model, "drowsiness_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("\n Model Training Complete & Saved!")
import joblib
import pandas as pd


model = joblib.load("drowsiness_model.pkl")
le = joblib.load("label_encoder.pkl")

sample = pd.DataFrame([{
    "perclos": 32.5,
    "ear": 0.14,
    "mar": 0.72,
    "blink_count": 2,
    "blink_rate_60s": 5,
    "head_rotation": 8,
    "eyes_closed_score": 0.32,
    "is_eye_closed_now": 1,
    "is_yawning_now": 1,
    "head_turn_severity": 8,
    "fatigue_score": 52.1,
    "blink_abnormality": 1,
    "distraction_flag": 0,
    "drowsy_flag": 1,
    "yawn_intensity": 0.22,
    "blink_slowing": 1
}])


pred = model.predict(sample)
action = le.inverse_transform(pred)

print(" AI Recommended Action:", action[0])


import json
import joblib
import numpy as np
from google import genai  
from gtts import gTTS
import os
import time



client = genai.Client(api_key=GEMINI_API_KEY)


artifact = joblib.load("drowsiness_model_with_features.pkl")

if isinstance(artifact, dict):
    clf = artifact.get("model")
    FEATURES = artifact.get("features", [])
else:
    clf = artifact
    FEATURES = [
        "perclos", "ear", "mar", "blink_count", "blink_rate_60s", "head_rotation",
        "eyes_closed_score", "is_eye_closed_now", "is_yawning_now", "head_turn_severity",
        "fatigue_score", "blink_abnormality", "distraction_flag", "drowsy_flag",
        "yawn_intensity", "blink_slowing"
    ]

le = joblib.load("label_encoder.pkl")

FALLBACK = {
    "breathing_offer": {
        "severity": "moderate",
        "audio_text": "Would you like a 2-minute breathing exercise?",
        "actions": [
            {"id": "breath", "label": "2-min breathing", "details": "Inhale 4s hold 4s exhale 6s", "duration_seconds": 120}
        ],
        "instructions": "Inhale 4s — hold 4s — exhale 6s."
    },
    "music_offer": {
        "severity": "moderate",
        "audio_text": "Shall I play a short energizing song?",
        "actions": [
            {"id": "music", "label": "Play music", "details": "Play 1-2 minute energizing track", "duration_seconds": 90}
        ],
        "instructions": ""
    },
    "nudge": {
        "severity": "low",
        "audio_text": "Quick reminder — please focus on the road.",
        "actions": [
            {"id": "refocus", "label": "Refocus", "details": "Look forward and hold wheel for 5s", "duration_seconds": 5}
        ],
        "instructions": ""
    },
    "no_action": {
        "severity": "low",
        "audio_text": "All good — keep driving safely.",
        "actions": [],
        "instructions": ""
    },
    "pull_over": {
        "severity": "high",
        "audio_text": "Warning — please pull over safely at the next stop.",
        "actions": [
            {"id": "pull_over", "label": "Pull over", "details": "Slow down, signal, and stop safely", "duration_seconds": 0}
        ],
        "instructions": "Slow down, signal, and stop safely."
    }
}


def predict_label_confidence_from_metrics(metrics):
    X = np.array([[float(metrics.get(f, 0)) for f in FEATURES]])
    probs = clf.predict_proba(X)[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    label = le.inverse_transform([idx])[0]
    return label, conf


def safety_override(metrics, label):
    if float(metrics.get("perclos", 0)) > 35 or float(metrics.get("ear", 1.0)) < 0.12:
        return "pull_over"
    return label


def ask_gemini(label, confidence, metrics):
    prompt = f"""
You are an in-vehicle safety assistant with a warm, calm, supportive tone.
Return ONLY a valid JSON object. No explanation, no extra text.

Output format (strict):
{{
  "severity": "low | moderate | high",
  "audio_text": "One short friendly sentence under 25 words",
  "instructions": "2-3 line helpful guidance on what to do next"
}}

Rules:
- Never repeat the same message twice. Always vary wording.
- Keep tone encouraging, gentle, non-judgmental, and caring.
- Based on severity give suitable suggestions:

HIGH → Ask to pull over safely + steps (3 steps max), plus offer breathing or stretch
MODERATE → Nudge + suggest stretch, breathing, posture fix, or “play music”
LOW → Light reminder, blink reset, hydrate, or offer music casually

Guidelines for instruction content:
- If suggesting pull over → include 3 quick steps
- If offering breathing → give 2–3 step breathing guide
- If suggesting stretch → give 2–3 simple stretch steps
- If offering music → only write: “play music”
- Never include timing or code, always natural language
- Keep audio_text < 25 words

Context:
Label: {label}
Confidence: {confidence:.2f}
Metrics: {json.dumps(metrics)}

Return ONLY JSON. No surrounding text.
"""

    try:
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt
        )

        text = response.text.strip()
        try:
            return json.loads(text)
        except:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(text[start:end + 1])
            return None

    except Exception as e:
        print(" Gemini Error:", e)
        return None


def speak_text(text, filename="output.mp3"):
    """Convert text to speech, save as MP3, and play."""
    if not text:
        return
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(filename)

   
    if os.name == "nt":     
        os.system(f"start {filename}")
    elif os.name == "posix":
        os.system(f"xdg-open {filename}")

  
    time.sleep(2)


if __name__ == "__main__":
    sample_metrics = {
        "perclos": 38.2,
        "ear": 0.11,
        "mar": 0.65,
        "blink_count": 22,
        "blink_rate_60s": 18,
        "fatigue_score": 0.82,
        "yawn_intensity": 0.3
    }

    label, conf = predict_label_confidence_from_metrics(sample_metrics)
    label = safety_override(sample_metrics, label)

    intervention = ask_gemini(label, conf, sample_metrics)
    if intervention is None:
        intervention = FALLBACK.get(label, FALLBACK["no_action"])

    print("\n MODEL:", label, conf)
    print("\n GEMINI INTERVENTION:")
    print(json.dumps(intervention, indent=2))

 
    speak_text(intervention.get("audio_text"), "audio_text.mp3")
    speak_text(intervention.get("instructions"), "instructions.mp3")
