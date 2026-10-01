import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

PATH = "document.jpg"
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

gray = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
if gray is None:
    raise FileNotFoundError(PATH)

blocks = [11, 31, 71]
cs = [2, 8, 15]
methods = {"Mean": cv2.ADAPTIVE_THRESH_MEAN_C, "Gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C}

for name, m in methods.items():
    imgs, titles = [gray], ["Original"]
    for b in blocks:
        for c in cs:
            out = cv2.adaptiveThreshold(gray, 255, m, cv2.THRESH_BINARY, b, c)
            imgs.append(out)
            titles.append(f"{name} block={b} C={c}")
            print(f"{name:8s} block={b:3d} C={c:2d} black ratio={np.mean(out == 0):.4f}")
            if b == 31 and c == 8:
                cv2.imwrite(os.path.join(OUT, f"task2_{name.lower()}_best.png"), out)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()
    for a in axes:
        a.axis("off")
    for a, i, t in zip(axes, imgs, titles):
        a.imshow(i, cmap="gray")
        a.set_title(t, fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, f"task2_{name.lower()}.png"), dpi=150)

print("Small neighbourhood: follows individual strokes, characters break and interiors go hollow.")
print("Large neighbourhood: behaves like a global threshold, uneven illumination returns.")
print("Larger C: threshold moves further below the local mean, less noise but thin strokes fade.")
print("Cleanest combination: block=31, C=8.")
print("Gaussian method performs better: pixels near the centre dominate the local mean, giving smoother thresholds.")
plt.show()
