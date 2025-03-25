import cv2 as cv
import numpy as np

def rotate_image(image_path, angle, save_path):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"File '{image_path}' could not be read. Check if the path is correct.")
    
    rows, cols = img.shape
    
    # Compute the rotation matrix
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, 1.0)
    
    # Calculate the new bounding dimensions
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_width = int((rows * sin) + (cols * cos))
    new_height = int((rows * cos) + (cols * sin))
    
    # Adjust the rotation matrix to take into account translation
    M[0, 2] += (new_width / 2) - (cols / 2)
    M[1, 2] += (new_height / 2) - (rows / 2)
    
    # Perform the rotation
    rotated_img = cv.warpAffine(img, M, (new_width, new_height))
    
    # Save the rotated image
    cv.imwrite(save_path, rotated_img)
    
    return rotated_img

# Example usage
image_path = 'Lab_1/Images/Sample_1.png'
save_path = 'Lab_1/Task_1/output.png'
rotated = rotate_image(image_path, 45, save_path)
cv.imshow('Rotated Image', rotated)
cv.waitKey(0)
cv.destroyAllWindows()
