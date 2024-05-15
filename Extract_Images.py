import cv2
import os

def save_frames(video_path, output_dir, target_resolution=(640, 360)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open the video file.")
        return

    # Initialize frame count
    frame_count = 0

    # Read video frames
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Check if frame reading was successful
        if not ret:
            break

        # Resize the frame to the target resolution
        frame_resized = cv2.resize(frame, target_resolution)

        # Save the resized frame as an image
        frame_name = f"frame_{frame_count:04d}.jpg"  # Frame name with zero-padding
        frame_path = os.path.join(output_dir, frame_name)
        cv2.imwrite(frame_path, frame_resized)

        # Increment frame count
        frame_count += 1

        # Print progress
        print(f"Frame {frame_count} saved.")

    # Release the video capture object
    cap.release()

    print("Video frames saved successfully.")

# Example usage
video_path = "D:/Bloody Roar 2 project/gameplay video/videoplayback.mp4"
output_dir = "D:/Bloody Roar 2 project/Pose and Tracking Dataset/Training Set/Images"
target_resolution = (256, 144)  # Set the target resolution (width, height)
save_frames(video_path, output_dir, target_resolution)
