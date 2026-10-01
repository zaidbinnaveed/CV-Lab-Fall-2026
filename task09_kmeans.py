import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

PATH = "wildlife.jpg"
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

bgr = cv2.imread(PATH)
if bgr is None:
    raise FileNotFoundError(PATH)

data = bgr.reshape((-1, 3)).astype(np.float32)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
orig_colors = len(np.unique(bgr.reshape(-1, 3), axis=0))

imgs, titles = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)], ["Original"]
for k in (2, 4, 6):
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    out = centers[labels.flatten()].reshape(bgr.shape)
    mse = np.mean((bgr.astype(np.float32) - out.astype(np.float32)) ** 2)
    print(f"K={k}: clusters={len(centers)} unique colors {orig_colors} -> {len(np.unique(out.reshape(-1, 3), axis=0))} MSE vs original={mse:.1f}")
    for i, c in enumerate(centers):
        print(f"  cluster {i}: BGR={c.tolist()} share={np.mean(labels.flatten() == i) * 100:.1f}%")
    imgs.append(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    titles.append(f"K={k}")
    cv2.imwrite(os.path.join(OUT, f"task9_k{k}.png"), out)

fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
for a, i, t in zip(axes, imgs, titles):
    a.imshow(i)
    a.set_title(t)
    a.axis("off")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "task9_output.png"), dpi=150)
plt.show()
