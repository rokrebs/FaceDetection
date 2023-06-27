import cv2
import dlib
import os
import time

# Initialize the dlib face detector
detector = dlib.get_frontal_face_detector()

# Initialize the video capture
video_capture = cv2.VideoCapture(0)

# Define the video codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_file = 'results/dlib_output/video_output.mp4'
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
    faces = detector(gray)

    # Iterate over the detected faces
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (51, 255, 255), 2)

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
