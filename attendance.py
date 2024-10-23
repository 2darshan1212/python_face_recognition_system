from flask import Flask, jsonify
from flask_cors import CORS
import cv2
import face_recognition
import numpy as np
import os
import pymongo
from datetime import datetime, timedelta
import threading
import logging

app = Flask(__name__)
CORS(app, origins=['http://localhost:5173'])

# MongoDB connection
MONGODB_URI = "mongodb+srv://darshanmisaal1212:darshan1212@cluster0.lwqbl.mongodb.net/attendance?retryWrites=true&w=majority&appName=Cluster0"
client = pymongo.MongoClient(MONGODB_URI)
db = client['attendance_system']

# Known face data
known_face_encodings = []
known_face_names = []
known_face_details = {}
last_attendance_time = {}
video_capture = None

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load known faces from the database and images folder
# @app.route('/load_known_faces', methods=['GET'])
# def load_known_faces():
#     global known_face_encodings, known_face_names, known_face_details
#     logging.info("Loading known faces...")

#     known_face_encodings.clear()
#     known_face_names.clear()

#     for filename in os.listdir('images'):
#         if filename.lower().endswith((".jpg", ".png")):
#             name = os.path.splitext(filename)[0]
#             img_path = os.path.join('images', filename)
#             img = face_recognition.load_image_file(img_path)
#             encodings = face_recognition.face_encodings(img)

#             if encodings:
#                 known_face_encodings.append(encodings[0])
#                 known_face_names.append(name)

#                 # Fetch details from MongoDB
#                 user_data = db['users'].find_one({"name": name})
#                 if user_data:
#                     known_face_details[name] = {
#                         "enrollment_number": user_data['enrollment_number'],
#                         "semester": user_data['semester']
#                     }

#     logging.info(f"Loaded {len(known_face_names)} faces.")
#     return jsonify({"message": "Known faces loaded"}), 200


@app.route('/load_known_faces', methods=['GET'])
def load_known_faces():
    global known_face_encodings, known_face_names, known_face_details
    logging.info("Loading known faces...")

    known_face_encodings.clear()
    known_face_names.clear()

    for filename in os.listdir('images'):
        if filename.lower().endswith((".jpg", ".png")):
            name = os.path.splitext(filename)[0]
            img_path = os.path.join('images', filename)
            try:
                img = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(img)

                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(name)

                    # Fetch details from MongoDB
                    user_data = db['users'].find_one({"name": name})
                    if user_data:
                        known_face_details[name] = {
                            "enrollment_number": user_data['enrollment_number'],
                            "semester": user_data['semester']
                        }
                else:
                    logging.warning(f"No encodings found for {name}.")
            except Exception as e:
                logging.error(f"Error processing {filename}: {e}")

    logging.info(f"Loaded {len(known_face_names)} faces.")
    return jsonify({"message": "Known faces loaded"}), 200



# Update attendance HTML file
def update_attendance_html(attendance_records):
    date_today = datetime.now().strftime("%Y-%m-%d")
    html_filename = f"{date_today}_attendance.html"

    if not os.path.exists(html_filename):
        with open(html_filename, 'w') as file:
            file.write("<html><head><title>Attendance</title></head><body>")
            file.write(f"<h2>Attendance for {date_today}</h2><table border='1'><tr><th>Name</th><th>Enrollment Number</th><th>Semester</th><th>Status</th><th>Timestamp</th></tr>")

    with open(html_filename, 'a') as file:
        for name, timestamp in attendance_records:
            details = known_face_details.get(name, {"enrollment_number": "N/A", "semester": "N/A"})
            file.write(f"<tr><td>{name}</td><td>{details['enrollment_number']}</td><td>{details['semester']}</td><td>Present</td><td>{timestamp}</td></tr>")

    logging.info(f"Updated HTML file: {html_filename}")

# Mark attendance for recognized user
def mark_attendance(name):
    now = datetime.now()
    date_today = now.strftime("%Y-%m-%d")
    time_only = now.strftime("%H:%M:%S")

    # Avoid marking attendance twice within 30 minutes for the same user
    if name in last_attendance_time and now - last_attendance_time[name] < timedelta(minutes=30):
        return

    last_attendance_time[name] = now
    collection = db[name]

    if not collection.find_one({"date": date_today}):
        attendance_record = {"date": date_today, "timestamp": time_only, "status": "Present"}
        collection.insert_one(attendance_record)
        logging.info(f"Attendance marked for {name} at {time_only}.")
        update_attendance_html([(name, time_only)])

# Advanced Preprocessing: contrast, denoise, sharpen, etc.
def preprocess_frame(frame):
    # Convert to grayscale and apply histogram equalization
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized_frame = cv2.equalizeHist(gray_frame)
    # Apply Gaussian blur to reduce noise
    blurred_frame = cv2.GaussianBlur(equalized_frame, (5, 5), 0)
    return cv2.cvtColor(blurred_frame, cv2.COLOR_GRAY2BGR)

# Face recognition process in a separate thread
def recognize_faces_in_thread():
    global video_capture

    # Check if camera is opened
    if video_capture is None or not video_capture.isOpened():
        video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        logging.error("Error: Could not open webcam.")
        return

    process_this_frame = True

    while True:
        ret, frame = video_capture.read()
        if not ret:
            logging.error("Error: Failed to capture frame.")
            break

        # Preprocessing for better accuracy
        frame = preprocess_frame(frame)

        # Process alternate frames for efficiency
        if process_this_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                if len(known_face_encodings) > 0:
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    # Set a dynamic threshold
                    threshold = 0.6  # You can adjust this value for better accuracy

                    if face_distances[best_match_index] < threshold:
                     name = known_face_names[best_match_index]
                    else:
                     name = "Unknown"
                else:
                     name = "Unknown"

                face_names.append(name)

                if name != "Unknown":
                    mark_attendance(name)

            # Draw a rectangle around each face and display the name
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left + 6, bottom + 25), cv2.FONT_HERSHEY_DUPLEX, 0.75, color, 1)

        process_this_frame = not process_this_frame  # Process alternate frames

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

@app.route('/start_recognition', methods=['POST'])
def start_recognition():
    thread = threading.Thread(target=recognize_faces_in_thread)
    thread.start()
    return jsonify({"message": "Face recognition started in a new thread"}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000) 