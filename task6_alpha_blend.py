import cv2
import matplotlib.pyplot as plt

image_path = 'image.jpg'
caption = 'My Caption'
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found.")
else:
    h, w = image.shape[:2]
    overlay = image.copy()
    band_start = int(h * 0.8)
    cv2.rectangle(overlay, (0, band_start), (w, h), (255, 0, 0), thickness=-1)

    blended = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)
    cv2.putText(blended, caption, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.axis('off')
    plt.imshow(blended_rgb)
    plt.show()
