import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

PATH = "coins.png"
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

bgr = cv2.imread(PATH)
if bgr is None:
    raise FileNotFoundError(PATH)

gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
sure_bg = cv2.dilate(opening, kernel, iterations=3)
dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
n, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0
markers = cv2.watershed(bgr.copy(), markers)
result = bgr.copy()
result[markers == -1] = [0, 0, 255]

print(f"Foreground markers={n - 1} separated regions={len(np.unique(markers[markers > 1]))}")

dist_vis = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
imgs = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), thresh, sure_bg, dist_vis, sure_fg, unknown, cv2.cvtColor(result, cv2.COLOR_BGR2RGB)]
titles = ["Original", "Threshold", "Sure Background", "Distance Transform", "Sure Foreground", "Unknown Region", "Final Watershed"]
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()
for a in axes:
    a.axis("off")
for a, i, t in zip(axes, imgs, titles):
    a.imshow(i, cmap="gray" if i.ndim == 2 else None)
    a.set_title(t)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "task7_output.png"), dpi=150)
cv2.imwrite(os.path.join(OUT, "task7_watershed.png"), result)
plt.show()
