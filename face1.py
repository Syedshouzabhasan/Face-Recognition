import face_recognition
import cv2
import csv
from datetime import datetime
import numpy as np

# Use a video file as input instead of the webcam
video_capture = cv2.VideoCapture(0)  # Replace with your video file path

# Check if the video file is accessible
if not video_capture.isOpened():
    print("Error: Could not open video file.")
    exit()

# Load known faces and their encodings
haider_image = face_recognition.load_image_file("/workspaces/face/photo/haider (1).jpeg")
haider_encode = face_recognition.face_encodings(haider_image)[0]

sameer_image = face_recognition.load_image_file("/workspaces/face/photo/sameer (1).jpeg")
sameer_encode = face_recognition.face_encodings(sameer_image)[0]

known_face = [haider_encode, sameer_encode]
known_face_name = ["haider", "sameer"]

student = known_face_name.copy()

# Initialize variables
face_locations = []
face_encodings = []
face_names = []
process_frame = True

# Get the current date for the attendance file
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Open CSV file to log attendance
with open(current_date + '.csv', 'w+', newline='') as f:
    lnwrite = csv.writer(f)

    while True:
        # Capture frame-by-frame
        _, frame = video_capture.read()
        
        # Break the loop if the video ends
        if frame is None:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]  # Convert BGR to RGB

        if process_frame:
            # Find all face locations and encodings in the frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_name[best_match_index]

                face_names.append(name)

                # Mark attendance
                if name in student:
                    student.remove(name)
                    print(f"Marked present: {name}")
                    current_time = now.strftime("%H:%M:%S")
                    lnwrite.writerow([name, current_time])

        process_frame = not process_frame  # Alternate frame processing for performance

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Label the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Display the resulting frame
        cv2.imshow("Attendance System", frame)

        # Break the loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
