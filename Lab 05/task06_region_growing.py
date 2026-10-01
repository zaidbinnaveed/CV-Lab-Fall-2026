import os
from collections import deque
import cv2
import numpy as np
import matplotlib.pyplot as plt

PATH = "brain_mri.png"
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

gray = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
if gray is None:
    raise FileNotFoundError(PATH)
h, w = gray.shape


def grow(img, seed, thr):
    mask = np.zeros(img.shape, np.uint8)
    ref = int(img[seed[1], seed[0]])
    q = deque([seed])
    mask[seed[1], seed[0]] = 255
    while q:
        x, y = q.popleft()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < img.shape[1] and 0 <= ny < img.shape[0] and mask[ny, nx] == 0:
                    if abs(int(img[ny, nx]) - ref) <= thr:
                        mask[ny, nx] = 255
                        q.append((nx, ny))
    return mask


seeds = [(w // 2, h // 2), (w // 3, h // 3)]
thrs = [5, 15, 30]
imgs, titles = [gray], ["Original"]
for s in seeds:
    for t in thrs:
        m = grow(gray, s, t)
        imgs.append(m)
        titles.append(f"seed={s} thr={t}")
        print(f"seed={s} thr={t}: region pixels={np.count_nonzero(m)}")
        cv2.imwrite(os.path.join(OUT, f"task6_seed{s[0]}_{s[1]}_thr{t}.png"), m)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()
for a in axes:
    a.axis("off")
for a, i, t in zip(axes, imgs, titles):
    a.imshow(i, cmap="gray")
    a.set_title(t, fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "task6_output.png"), dpi=150)
print("Changing the seed changes the reference intensity and the starting connected area; region growing only reaches pixels connected to the seed that satisfy the intensity condition, so a seed in a different tissue yields a different region.")
plt.show()
