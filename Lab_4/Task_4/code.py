# Hough Line Transform for Line Detection
import cv2
import os
import numpy as np

# Load an image from file
image_path = r"Lab_4\Images\Sample_4.jpg"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # 5x5 kernel

# Use the Canny edge detector to find edges
edges = cv2.Canny(blurred, 50, 150)  # Gradient values above 150 are strong edges

# Hough Transform to detect lines
lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

# Draw the lines on the original image
if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red lines with thickness 2

# Ensure the output directory exists
output_dir = r'Lab_4\Task_4'
os.makedirs(output_dir, exist_ok=True)

# Save the result
output_path = os.path.join(output_dir, 'Hough_transform.jpg')
cv2.imwrite(output_path, image)

print(f"Saved: {output_path}")