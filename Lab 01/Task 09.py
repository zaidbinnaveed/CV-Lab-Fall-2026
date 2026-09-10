import cv2
import pandas as pd

image_path = 'image.jpg'
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found.")
else:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    red = image_rgb[:, :, 0].flatten()
    green = image_rgb[:, :, 1].flatten()
    blue = image_rgb[:, :, 2].flatten()

    df = pd.DataFrame({"Red": red, "Green": green, "Blue": blue})
    print(df.describe())
