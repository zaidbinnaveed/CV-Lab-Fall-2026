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

ts = [80, 120, 160]
glob = [cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)[1] for t in ts]
block, c = 31, 10
adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, c)

h, w = gray.shape
for t, m in zip(ts, glob):
    print(f"Global T={t}: black ratio left={np.mean(m[:, :w // 2] == 0):.3f} right={np.mean(m[:, w // 2:] == 0):.3f}")
print(f"Adaptive Gaussian block={block} C={c}: black ratio left={np.mean(adaptive[:, :w // 2] == 0):.3f} right={np.mean(adaptive[:, w // 2:] == 0):.3f}")
print("Parameters: T=80/120/160 cover dark, medium and bright page levels; block=31 spans a few characters; C=10 suppresses paper noise.")
print("Failure regions: the dark side loses characters or turns black at low T; the bright side turns white or merges at high T.")

images = [gray] + glob + [adaptive]
titles = ["Original", f"Global T1={ts[0]}", f"Global T2={ts[1]}", f"Global T3={ts[2]}", "Adaptive"]
fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
for a, i, t in zip(axes, images, titles):
    a.imshow(i, cmap="gray")
    a.set_title(t)
    a.axis("off")
fig.text(0.5, 0.02, "Why does a single threshold struggle when illumination changes? Because one fixed intensity cannot separate text from paper when the paper's own brightness varies across the image.", ha="center", fontsize=9)
plt.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(os.path.join(OUT, "task1_output.png"), dpi=150)
cv2.imwrite(os.path.join(OUT, "task1_adaptive.png"), adaptive)
plt.show()
