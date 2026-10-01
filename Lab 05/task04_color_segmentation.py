import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

PATH = "yellow_car.jpg"
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

bgr = cv2.imread(PATH)
if bgr is None:
    raise FileNotFoundError(PATH)
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

lo_r, up_r = np.array([26, 220, 220]), np.array([30, 255, 255])
lo_f, up_f = np.array([15, 80, 80]), np.array([35, 255, 255])

mask_r = cv2.inRange(hsv, lo_r, up_r)
mask_f = cv2.inRange(hsv, lo_f, up_f)
ext_r = cv2.bitwise_and(bgr, bgr, mask=mask_r)
ext_f = cv2.bitwise_and(bgr, bgr, mask=mask_f)

print(f"Restrictive mask pixels={np.count_nonzero(mask_r)} | final mask pixels={np.count_nonzero(mask_f)}")
print("The restrictive mask fails because the car's hue, saturation and value vary with shading and reflections; narrow bounds keep only a few ideal pixels and drop shaded or highlighted parts.")

imgs = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), mask_r, cv2.cvtColor(ext_r, cv2.COLOR_BGR2RGB), mask_f, cv2.cvtColor(ext_f, cv2.COLOR_BGR2RGB)]
titles = ["Original", "Restrictive mask", "Restrictive extraction", "Final mask", "Final extraction"]
fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
for a, i, t in zip(axes, imgs, titles):
    a.imshow(i, cmap="gray" if i.ndim == 2 else None)
    a.set_title(t)
    a.axis("off")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "task4_output.png"), dpi=150)
cv2.imwrite(os.path.join(OUT, "task4_final_mask.png"), mask_f)
plt.show()
