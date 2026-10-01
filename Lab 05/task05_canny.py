import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

PATH = "part.jpg"
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

gray = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
if gray is None:
    raise FileNotFoundError(PATH)
blur = cv2.GaussianBlur(gray, (5, 5), 1.4)

pairs = [(50, 100), (50, 150), (50, 250)]
edges = [cv2.Canny(blur, lo, hi) for lo, hi in pairs]

strong = edges[2] > 0
weak = (edges[0] > 0) & ~strong
for (lo, hi), e in zip(pairs, edges):
    print(f"Canny low={lo} high={hi}: edge pixels={np.count_nonzero(e)}")
print(f"Strong edges (survive high=250): {np.count_nonzero(strong)} px")
print(f"Weak edges (only in low=50/high=100): {np.count_nonzero(weak)} px")
print("Missing edges: boundary segments present in result 1 but gone in result 3.")
print("Unwanted edges: texture and noise pixels in result 1 that are not on the part boundary.")
print("Raising the high threshold with a fixed low threshold removes strong seed edges, so hysteresis connects fewer weak edges: output is cleaner but boundaries can break.")

imgs = [gray] + edges
titles = ["Original"] + [f"Edge Result {i + 1} ({lo},{hi})" for i, (lo, hi) in enumerate(pairs)]
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
for a, i, t in zip(axes, imgs, titles):
    a.imshow(i, cmap="gray")
    a.set_title(t)
    a.axis("off")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "task5_output.png"), dpi=150)
plt.show()
