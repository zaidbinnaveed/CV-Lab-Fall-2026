import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path1 = 'image1.jpg'
image_path2 = 'image2.jpg'

img1 = cv2.imread(image_path1)
img2 = cv2.imread(image_path2)

if img1 is None or img2 is None:
    print("Error: Image not found.")
else:
    size = (500, 500)
    img1 = cv2.resize(img1, size)
    img2 = cv2.resize(img2, size)

    mask = np.zeros(size, dtype=np.uint8)
    cv2.circle(mask, (250, 250), 200, 255, thickness=-1)

    fg = cv2.bitwise_and(img1, img1, mask=mask)
    inverse_mask = cv2.bitwise_not(mask)
    bg = cv2.bitwise_and(img2, img2, mask=inverse_mask)
    result = cv2.bitwise_or(fg, bg)

    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(result_rgb)
    plt.show()
