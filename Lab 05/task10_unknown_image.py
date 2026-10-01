import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

PATH = "unknown.jpg"
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

bgr = cv2.imread(PATH)
if bgr is None:
    raise FileNotFoundError(PATH)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)


def kmeans_dark(img, k):
    data = img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    vis = centers[labels.flatten()].reshape(img.shape)
    dark = np.argmin(centers.astype(np.float32).mean(axis=1))
    mask = (labels.reshape(img.shape[:2]) == dark).astype(np.uint8) * 255
    return vis, mask


def fg_ratio(m):
    return float(np.mean(m > 0))


def fragments(m):
    n, _, stats = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8))
    if n <= 1:
        return 1.0
    return float(np.sum(stats[1:, cv2.CC_STAT_AREA] < 20)) / (n - 1)


t_otsu, m1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
m2 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
vis3, m3 = kmeans_dark(bgr, 3)

sens1 = [fg_ratio(cv2.threshold(blur, t, 255, cv2.THRESH_BINARY_INV)[1]) for t in (t_otsu - 20, t_otsu, t_otsu + 20)]
sens2 = [fg_ratio(cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, c)) for c in (2, 10, 20)]
sens3 = [fg_ratio(kmeans_dark(bgr, k)[1]) for k in (2, 3, 5)]

report = {
    "Otsu": ("bimodal histogram and uniform illumination", m1, "threshold", np.ptp(sens1)),
    "Adaptive Gaussian": ("illumination varies slowly and the object is locally darker than its surroundings", m2, "C", np.ptp(sens2)),
    "K-Means": ("object and background form separate colour clusters", m3, "K", np.ptp(sens3)),
}

best, best_frag = None, 2.0
for name, (assumption, mask, param, spread) in report.items():
    frag = fragments(mask)
    print(f"{name}: assumes {assumption}")
    print(f"  foreground ratio={fg_ratio(mask):.3f} fragment ratio={frag:.3f}")
    print(f"  most influential parameter={param} (foreground ratio spread={spread:.3f})")
    if frag < best_frag:
        best, best_frag = name, frag

print("Incorrectly segmented parts: shadows, texture or background areas whose intensity or colour matches the object appear as foreground; check the comparison figure.")
print(f"Conclusion: {best} produced the most coherent regions (lowest fragment ratio {best_frag:.3f}), indicating that the image property it assumes holds for this image.")

imgs = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), m1, m2, cv2.cvtColor(vis3, cv2.COLOR_BGR2RGB)]
titles = ["Original", "Method 1: Otsu", "Method 2: Adaptive Gaussian", "Method 3: K-Means (K=3)"]
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
for a, i, t in zip(axes, imgs, titles):
    a.imshow(i, cmap="gray" if i.ndim == 2 else None)
    a.set_title(t)
    a.axis("off")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "task10_output.png"), dpi=150)
plt.show()
