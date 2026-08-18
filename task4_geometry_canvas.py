import cv2
import numpy as np
import matplotlib.pyplot as plt

canvas = np.zeros((800, 800, 3), dtype=np.uint8)
center = (400, 400)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
radii = [350, 280, 210, 140, 70]

for radius, color in zip(radii, colors):
    cv2.circle(canvas, center, radius, color, thickness=8)

top_left = (center[0] - radii[0], center[1] - radii[0])
bottom_right = (center[0] + radii[0], center[1] + radii[0])
cv2.rectangle(canvas, top_left, bottom_right, (255, 255, 255), thickness=3)

print("Center coordinates:", center)

plt.figure(figsize=(6, 6))
plt.axis('off')
plt.imshow(canvas)
plt.show()
