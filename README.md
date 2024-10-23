import cv2
import face_recognition
import numpy as np
import os
import pymongo
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# Ensure the 'images' directory exists
if not os.path.exists('images'):
    os.makedirs('images')

# MongoDB connection string (replace with your actual credentials)
MONGODB_URI = "mongodb+srv://darshanmisaal1212:darshan1212@cluster0.lwqbl.mongodb.net/attendance?retryWrites=true&w=majority&appName=Cluster0"

# Connect to MongoDB
client = pymongo.MongoClient(MONGODB_URI)
db = client['attendance_system']

# Load known faces and their encodings
known_face_encodings = []
known_face_names = []
known_face_details = {}  # To store enrollment and semester details
last_attendance_time = {}

def load_known_faces():
    global known_face_encodings, known_face_names, known_face_details
    print("Loading known faces...")

    # Iterate through 'images' folder to load known faces
    for filename in os.listdir('images'):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            name = os.path.splitext(filename)[0]
            img_path = os.path.join('images', filename)
            img = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(img)

            if len(encodings) > 0:
                encoding = encodings[0]
                known_face_encodings.append(encoding)
                known_face_names.append(name)
                # Load enrollment number and semester from the database
                user_data = db['attendance_system'].find_one({"name": name})
                if user_data:
                    known_face_details[name] = {
                        "enrollment_number": user_data['enrollment_number'],
                        "semester": user_data['semester']
                    }
    print(f"Loaded {len(known_face_names)} faces.")

# Function to generate or update an HTML file for the current date
def update_attendance_html(attendance_records):
    date_today = datetime.now().strftime("%Y-%m-%d")
    html_filename = f"{date_today}_attendance.html"
    
    # Check if the file exists; if not, create it with a table structure
    if not os.path.exists(html_filename):
        with open(html_filename, 'w') as file:
            file.write("<html>\n<head><title>Attendance</title></head>\n<body>\n")
            file.write(f"<h2>Attendance for {date_today}</h2>\n")
            file.write("<table border='1'>\n<tr><th>Name</th><th>Enrollment Number</th><th>Semester</th><th>Status</th><th>Timestamp</th></tr>\n")
    
    # Append the new attendance records to the table
    with open(html_filename, 'a') as file:
        for name, timestamp in attendance_records:
            enrollment_number = known_face_details.get(name, {}).get("enrollment_number", "N/A")
            semester = known_face_details.get(name, {}).get("semester", "N/A")
            file.write(f"<tr><td>{name}</td><td>{enrollment_number}</td><td>{semester}</td><td>Present</td><td>{timestamp}</td></tr>\n")
    
    print(f"Updated HTML file: {html_filename}")

# Mark attendance in MongoDB and update the HTML file every 30 minutes
def mark_attendance(name):
    now = datetime.now()
    date_today = now.strftime("%Y-%m-%d")
    time_only = now.strftime("%H:%M:%S")  # Format to show only time

    # Check if 30 minutes have passed since last attendance marking
    if name in last_attendance_time and now - last_attendance_time[name] < timedelta(minutes=30):
        print(f"Attendance already marked for {name} within the last 30 minutes.")
        return

    last_attendance_time[name] = now

    # Create a collection for each user (if not already exists)
    collection = db[name]

    # Check if attendance is already marked for today
    if collection.find_one({"date": date_today}):
        print(f"Attendance already marked for {name} on {date_today}.")
    else:
        attendance_record = {
            "date": date_today,
            "timestamp": time_only,  # Store only the time in the database
            "status": "Present"
        }
        collection.insert_one(attendance_record)
        print(f"Attendance marked for {name} at {time_only}.")
    
    # Update the HTML file with this attendance
    update_attendance_html([(name, time_only)])  # Pass only time for HTML


# Apply image preprocessing to improve detection in low-light or low-quality frames
def preprocess_image(frame):
    # Convert to grayscale and apply histogram equalization
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized_frame = cv2.equalizeHist(gray_frame)
    return equalized_frame

# Recognize faces from webcam feed with advanced handling for slow/laggy camera
def recognize_faces():
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return

    video_capture.set(3, 640)  # Set width to reduce processing
    video_capture.set(4, 480)  # Set height

    # Multi-threading pool for background tasks
    with ThreadPoolExecutor() as executor:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Failed to capture frame. Retrying...")
                continue

            # Preprocess the frame to enhance detection
            processed_frame = preprocess_image(frame)

            # Resize frame for faster processing
            small_frame = cv2.resize(processed_frame, (0, 0), fx=0.25, fy=0.25)

            # Convert to RGB format for face recognition
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_GRAY2RGB)

            # Find all face locations and encodings
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            face_names = []

            for face_encoding in face_encodings:
                # Compute the distance between the face encoding and known faces
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                # Confidence threshold for face recognition
                confidence_threshold = 0.45
                best_match_index = np.argmin(face_distances)

                if face_distances[best_match_index] < confidence_threshold:
                    name = known_face_names[best_match_index]
                else:
                    name = "Unknown"

                face_names.append(name)

                # Mark attendance for recognized faces (run this in the background)
                if name != "Unknown":
                    executor.submit(mark_attendance, name)

            # Display the video frame with bounding boxes and names
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Set color based on recognition
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

                # Draw the rectangle around the face
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

                # Display the name outside the rectangle
                cv2.putText(frame, name, (left + 6, top - 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 1)

            # Display the frame
            cv2.imshow('Video', frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the webcam and close windows
    video_capture.release()
    cv2.destroyAllWindows()

# Capture image from webcam and add new user
def add_user():
    name = input("Enter the name of the new user: ")
    enrollment_number = input("Enter the enrollment number: ")
    semester = input("Enter the semester: ")

    # Initialize webcam
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'c' to capture image.")

    while True:
        ret, frame = video_capture.read()

        if not ret:
            print("Error: Failed to capture image. Retrying...")
            continue

        # Display the frame
        cv2.imshow('Video', frame)

        # Wait for 'c' to capture image
        if cv2.waitKey(1) & 0xFF == ord('c'):
            print("Image captured.")
            img_user_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Save the captured image to the images folder
            img_path = os.path.join('images', f"{name}.jpg")
            cv2.imwrite(img_path, frame)

            # Add the new user details to the database
            user_data = {
                "name": name,
                "enrollment_number": enrollment_number,
                "semester": semester
            }
            db['users'].insert_one(user_data)
            print(f"New user added: {name}")

            break

    # Release the webcam and close windows
    video_capture.release()
    cv2.destroyAllWindows()

    # Reload known faces after adding the new user
    load_known_faces()

# Load known faces at the start
load_known_faces()

# Main menu
def main():
    while True:
        print("\nMenu:")
        print("1. Add new user")
        print("2. Start face recognition attendance system")
        print("3. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            add_user()
        elif choice == '2':
            recognize_faces()
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
