import cv2
import matplotlib.pyplot as plt

image_path = 'highres.jpg'
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found.")
else:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    blurred = cv2.GaussianBlur(image_rgb, (25, 25), 0)

    h, w = image_rgb.shape[:2]
    cy, cx = h // 2, w // 2
    half = 150

    roi_orig = image_rgb[cy - half:cy + half, cx - half:cx + half]
    roi_blur = blurred[cy - half:cy + half, cx - half:cx + half]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(roi_orig)
    axes[0].axis('off')
    axes[0].set_title("Original ROI", fontsize=14, color='black')
    axes[1].imshow(roi_blur)
    axes[1].axis('off')
    axes[1].set_title("Blurred ROI", fontsize=14, color='blue')
    plt.show()
