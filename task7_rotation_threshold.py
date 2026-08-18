import cv2
import matplotlib.pyplot as plt

image_path = 'document.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Image not found.")
else:
    _, global_thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    adaptive_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 2)

    h, w = adaptive_thresh.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 45, 0.8)
    rotated = cv2.warpAffine(adaptive_thresh, rotation_matrix, (w, h))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(global_thresh, cmap='gray')
    axes[0].axis('off')
    axes[0].set_title("Global Threshold")
    axes[1].imshow(adaptive_thresh, cmap='gray')
    axes[1].axis('off')
    axes[1].set_title("Adaptive Threshold")
    axes[2].imshow(rotated, cmap='gray')
    axes[2].axis('off')
    axes[2].set_title("Rotated Adaptive")
    plt.show()
