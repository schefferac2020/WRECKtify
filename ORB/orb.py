import cv2

def detect_orb_features(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initialize the ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # Draw keypoints on the image
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=0)

    # Display the result
    cv2.imshow("ORB Features", img_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Return keypoints and descriptors
    return keypoints, descriptors

# Example usage
image_path = "underwater.jpg"
keypoints, descriptors = detect_orb_features(image_path)
print(f"Number of ORB keypoints: {len(keypoints)}")