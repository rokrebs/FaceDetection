import cv2
import os
import time
# Load the Haar cascade XML file for face detection
cascade_path = 'models/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# Initialize the video capture
video_capture = cv2.VideoCapture(0)

# Define the video codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_file = 'results/viola_output/video_output.mp4'
output_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_fps = 30.0  
video_writer = cv2.VideoWriter(output_file, fourcc, output_fps, (output_width, output_height))

# Start the timer
start_time = time.time()

while True:
    # Read the current frame from the video capture
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (51, 255, 255), 10)

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Write the frame to the video file
    video_writer.write(frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop the timer
end_time = time.time()

# Calculate the total running time
total_time = end_time - start_time

# Release the video capture and video writer
video_capture.release()
video_writer.release()

# Destroy the OpenCV window
cv2.destroyAllWindows()

# Print the total running time
print(f"Total running time: {total_time} seconds")
