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


def run(ratio):
    _, sure_fg = cv2.threshold(dist, ratio * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    n, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(bgr.copy(), markers)
    result = bgr.copy()
    result[markers == -1] = [0, 0, 255]
    return n - 1, len(np.unique(markers[markers > 1])), result


ratios = [0.2, 0.4, 0.6, 0.8]
rows = []
imgs, titles = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)], ["Original"]
for i, r in enumerate(ratios, 1):
    markers_n, regions_n, res = run(r)
    rows.append((i, r, markers_n, regions_n))
    imgs.append(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    titles.append(f"Exp {i}: ratio={r}")
    cv2.imwrite(os.path.join(OUT, f"task8_exp{i}.png"), res)

print(f"{'Experiment':<12}{'Distance Threshold':<20}{'Foreground':<12}{'Regions':<10}")
for i, r, m, reg in rows:
    print(f"{i:<12}{r:<20}{m:<12}{reg:<10}")
print("Low ratio: markers touch and coins merge. High ratio: small coins lose markers or a coin is split. The most meaningful separation is in the middle range (about 0.4-0.6 of the maximum distance); confirm visually.")

fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
for a, i, t in zip(axes, imgs, titles):
    a.imshow(i)
    a.set_title(t)
    a.axis("off")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "task8_output.png"), dpi=150)
plt.show()
