import cv2
import os
# Define the directory containing images
image_folder = '/home/zhiychen/Desktop/AnimatableGaussians/test_results/185/Outer_precise_no_scale/training__cam_0000/batch_066360/vanilla/rgb_map'
video_name = 'animation_precise_no_scale.mp4'
# Get all the image files in the folder
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
# images = [img for img in os.listdir(image_folder) if img.endswith(“normal.png”) or img.endswith(“.jpg”)]
images.sort()  # Optional, to ensure the images are in the correct sequence
# Read the first image to get dimensions
first_image_path = os.path.join(image_folder, images[0])
frame = cv2.imread(first_image_path)
height, width, layers = frame.shape
# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(video_name, fourcc, 30, (width, height))
for image in images:
    image_path = os.path.join(image_folder, image)
    video.write(cv2.imread(image_path))
# Release the VideoWriter object
video.release()
cv2.destroyAllWindows()

print(f'Video {video_name} created successfully.')