import os
import cv2
import matplotlib.pyplot as plt

PATH = "coins.png"
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

gray = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
if gray is None:
    raise FileNotFoundError(PATH)

variants = {
    "Plain": gray,
    "Gaussian blur 5x5": cv2.GaussianBlur(gray, (5, 5), 0),
    "Contrast alpha=1.5": cv2.convertScaleAbs(gray, alpha=1.5, beta=0),
}

fig, axes = plt.subplots(len(variants), 3, figsize=(12, 4 * len(variants)))
for row, (name, img) in zip(axes, variants.items()):
    t, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"{name}: Otsu threshold = {t:.1f}")
    row[0].imshow(img, cmap="gray")
    row[0].set_title(f"Original ({name})")
    row[0].axis("off")
    row[1].hist(img.ravel(), bins=256, range=(0, 256))
    row[1].axvline(t, color="r")
    row[1].set_title(f"Histogram, Otsu T={t:.0f}")
    row[2].imshow(mask, cmap="gray")
    row[2].set_title("Otsu Binary Mask")
    row[2].axis("off")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "task3_output.png"), dpi=150)
print("Otsu is useful when the threshold is unknown beforehand: it automatically picks the value that maximises between-class variance of a bimodal histogram, so no manual tuning per image is needed.")
plt.show()
