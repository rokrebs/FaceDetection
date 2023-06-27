import cv2
import os
import time
import numpy as np

# Load the pre-trained model for face detection
model_path = 'models/res10_300x300_ssd_iter_140000.caffemodel'
config_path = 'models/deploy.prototxt'
net = cv2.dnn.readNetFromCaffe(config_path, model_path)

# Initialize the video capture
video_capture = cv2.VideoCapture(0)

# Define the video codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_file = 'results/dnn_output/video_output.mp4'
output_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_fps = 30.0  
video_writer = cv2.VideoWriter(output_file, fourcc, output_fps, (output_width, output_height))

# Start the timer
start_time = time.time()

while True:
    # Read the current frame from the video capture
    ret, frame = video_capture.read()

    # Resize the frame to a fixed width and height
    target_width = 300
    target_height = 300
    resized_frame = cv2.resize(frame, (target_width, target_height))

    # Create a blob from the resized frame
    blob = cv2.dnn.blobFromImage(resized_frame, 1.0, (target_width, target_height), (104.0, 177.0, 123.0))

    # Set the blob as input to the neural network
    net.setInput(blob)

    # Perform face detection
    detections = net.forward()

    # Iterate over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([target_width, target_height, target_width, target_height])
            (x, y, w, h) = box.astype(int)

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
