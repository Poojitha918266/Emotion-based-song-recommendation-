import cv2
from fer import FER
import random
import time
import os
from googleapiclient.discovery import build
import pyttsx3
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import matplotlib.pyplot as plt
from datetime import datetime

# Initialize the emotion detector
emotion_detector = FER()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Google API Key for YouTube
YOUTUBE_API_KEY = 'AIzaSyCF6mjOQDNa8OUtWvWd60Fz9gtzkOo-sQ8'
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Dictionaries for mappings
mappings = {
    "emotion_to_query": {
        'happy': 'happy song',
        'sad': 'sad song',
        'angry': 'angry song',
        'neutral': 'neutral background music',
        'surprised': 'surprise song',
        'fearful': 'fearful music',
        'disgusted': 'disgusted song',
        'confused': 'confused song',
        'bored': 'boredom relief song',
        'excited': 'exciting party song'
    },
    "emotion_to_volume": {
        'happy': 0.8,
        'sad': 0.5,
        'angry': 0.3,
        'neutral': 0.6,
        'surprised': 0.7,
        'fearful': 0.4,
        'disgusted': 0.5,
        'confused': 0.6,
        'bored': 0.6,
        'excited': 1.0
    },
    "time_of_day_to_query": {
        "morning": 'morning music',
        "afternoon": 'afternoon music',
        "evening": 'evening music',
        "night": 'night music'
    }
}

# Initialize pyttsx3 for voice feedback
engine = pyttsx3.init()

# Variables for time tracking
time_tracking = {
    "last_song_time": 0,
    "last_log_time": 0
}

# Mood tracking data
mood_data = {
    "mood_history": [],
    "timestamps": []
}

# Initialize pycaw for volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

def set_volume(emotion):
    """Set the system volume based on the detected emotion."""
    vol_level = mappings["emotion_to_volume"].get(emotion, 0.6)  # Default to 60%
    volume.SetMasterVolumeLevelScalar(vol_level, None)
    print(f"Set volume to {int(vol_level * 100)}% for emotion: {emotion}")

def search_youtube(query):
    """Search for a YouTube video based on the query."""
    search_response = youtube.search().list(
        q=query,
        type='video',
        part='id',
        maxResults=10
    ).execute()
    videos = search_response.get('items', [])
    if videos:
        return random.choice(videos)['id']['videoId']
    return None

def voice_feedback(text):
    """Provide voice feedback using pyttsx3."""
    engine.say(text)
    engine.runAndWait()

def show_mood_graph():
    """Display a graph showing mood trends over time."""
    if not mood_data["mood_history"]:
        print("No mood data available to display.")
        return
    plt.figure(figsize=(10, 5))
    plt.plot(mood_data["timestamps"], mood_data["mood_history"], marker='o', linestyle='-')
    plt.title("Mood Trends Over Time")
    plt.xlabel("Time")
    plt.ylabel("Detected Emotion")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def get_time_of_day():
    """Get the time of day and return a suggestion based on that."""
    current_hour = datetime.now().hour
    if 6 <= current_hour < 12:
        return mappings["time_of_day_to_query"]["morning"]
    elif 12 <= current_hour < 18:
        return mappings["time_of_day_to_query"]["afternoon"]
    elif 18 <= current_hour < 21:
        return mappings["time_of_day_to_query"]["evening"]
    else:
        return mappings["time_of_day_to_query"]["night"]

# Start webcam feed
cap = cv2.VideoCapture(0)

print("Press 'q' to quit. Press 'v' to visualize mood trends.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        emotions = emotion_detector.detect_emotions(face)

        if emotions:
            dominant_emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
            print(f"Detected Emotion: {dominant_emotion}")

            # Log emotions every 5 seconds
            current_time = time.time()
            if current_time - time_tracking["last_log_time"] >= 5:
                mood_data["mood_history"].append(dominant_emotion)
                mood_data["timestamps"].append(time.strftime("%H:%M:%S", time.localtime()))
                time_tracking["last_log_time"] = current_time

            # Set volume based on emotion
            set_volume(dominant_emotion)

            # Provide voice feedback
            voice_feedback(f"The detected emotion is {dominant_emotion}. Adjusting volume.")

            # Suggest a song every 1 minute
            if current_time - time_tracking["last_song_time"] >= 60:
                query = mappings["emotion_to_query"].get(dominant_emotion, 'background music')
                track_id = search_youtube(query)
                if track_id:
                    youtube_url = f'https://www.youtube.com/watch?v={track_id}'
                    print(f"Opening YouTube for {dominant_emotion}: {youtube_url}")
                    os.system(f'start {youtube_url}')
                    voice_feedback(f"Playing {dominant_emotion} music.")
                    time_tracking["last_song_time"] = current_time

            # Context-aware music recommendation based on time of day
            if current_time - time_tracking["last_song_time"] >= 60:  # Check for time gap
                time_of_day_query = get_time_of_day()
                track_id = search_youtube(time_of_day_query)
                if track_id:
                    youtube_url = f'https://www.youtube.com/watch?v={track_id}'
                    print(f"Opening YouTube for time of day recommendation: {youtube_url}")
                    os.system(f'start {youtube_url}')
                    voice_feedback(f"Playing {time_of_day_query}.")
                    time_tracking["last_song_time"] = current_time

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show frame
    cv2.imshow('Emotion-Based Music Recommendation', frame)

    # Quit or visualize graph
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Check for 'q' key to quit
        print("Exiting...")
        break
    elif key == ord('v'):  # Check for 'v' key to visualize mood trends
        print("Displaying mood trends...")
        show_mood_graph()

cap.release()
cv2.destroyAllWindows()
