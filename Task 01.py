import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def validate_environment():
    print("OpenCV version:", cv2.__version__)
    test = np.zeros((10, 10, 3), dtype=np.uint8)
    plt.imshow(test)
    plt.title("Validation")
    plt.axis('off')
    plt.show()


class GroceryManager:
    def __init__(self):
        self.items = {}

    def add_item(self, item, quantity, price):
        self.items[item] = {"quantity": quantity, "price": price}

    def remove_item(self, item):
        if item not in self.items:
            print(f"Error: '{item}' does not exist in the list.")
            return
        del self.items[item]

    def view_list(self):
        for item, details in self.items.items():
            print(f"{item}: Qty={details['quantity']}, Price={details['price']}")

    def calculate_total(self):
        return sum(d["quantity"] * d["price"] for d in self.items.values())


def run_grocery_manager_demo():
    gm = GroceryManager()
    gm.add_item("Apple", 4, 50)
    gm.add_item("Milk", 2, 180)
    gm.remove_item("Bread")
    gm.view_list()
    print("Total:", gm.calculate_total())


students = {
    "S001": {"Name": "Ali", "Major": "CS", "Grades": [85, 90, 78]},
    "S002": {"Name": "Sara", "Major": "EE", "Grades": [92, 88, 95]},
    "S003": {"Name": "Ahmed", "Major": "CS", "Grades": [70, 65, 80]},
}


def top_student(records):
    best_id = None
    best_avg = -1
    for sid, info in records.items():
        avg = sum(info["Grades"]) / len(info["Grades"])
        if avg > best_avg:
            best_avg = avg
            best_id = sid
    return records[best_id]["Name"]


def students_by_major(records, major):
    for sid, info in records.items():
        if info["Major"] == major:
            print(info["Name"])


def run_student_records_demo():
    print("Top student:", top_student(students))
    print("CS students:")
    students_by_major(students, "CS")


def task3_load_and_show(image_path="Sukuna.jpeg"):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.axis('off')
    plt.imshow(image_rgb)
    plt.title('King of Curses - Ryomen Sukuna', fontsize=16, color='darkred', pad=15)
    plt.show()


def task4_geometry_canvas():
    canvas = np.zeros((800, 800, 3), dtype=np.uint8)
    center = (400, 400)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    radii = [350, 280, 210, 140, 70]
    for radius, color in zip(radii, colors):
        cv2.circle(canvas, center, radius, color, thickness=8)
    top_left = (center[0] - radii[0], center[1] - radii[0])
    bottom_right = (center[0] + radii[0], center[1] + radii[0])
    cv2.rectangle(canvas, top_left, bottom_right, (255, 255, 255), thickness=3)
    print("Center coordinates:", center)
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(canvas)
    plt.show()


def task5_blur_roi(image_path="highres.jpg"):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    blurred = cv2.GaussianBlur(image_rgb, (25, 25), 0)
    h, w = image_rgb.shape[:2]
    cy, cx = h // 2, w // 2
    half = 150
    roi_orig = image_rgb[cy - half:cy + half, cx - half:cx + half]
    roi_blur = blurred[cy - half:cy + half, cx - half:cx + half]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(roi_orig)
    axes[0].axis('off')
    axes[0].set_title("Original ROI", fontsize=14, color='black')
    axes[1].imshow(roi_blur)
    axes[1].axis('off')
    axes[1].set_title("Blurred ROI", fontsize=14, color='blue')
    plt.show()


def task6_alpha_blend_caption(image_path="image.jpg", caption="My Caption"):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return
    h, w = image.shape[:2]
    overlay = image.copy()
    band_start = int(h * 0.8)
    cv2.rectangle(overlay, (0, band_start), (w, h), (255, 0, 0), thickness=-1)
    blended = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)
    cv2.putText(blended, caption, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.axis('off')
    plt.imshow(blended_rgb)
    plt.show()


def task7_rotation_thresholding(image_path="document.jpg"):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found.")
        return
    _, global_thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    adaptive_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 2)
    h, w = adaptive_thresh.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 45, 0.8)
    rotated = cv2.warpAffine(adaptive_thresh, rotation_matrix, (w, h))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(global_thresh, cmap='gray')
    axes[0].axis('off')
    axes[0].set_title("Global Threshold")
    axes[1].imshow(adaptive_thresh, cmap='gray')
    axes[1].axis('off')
    axes[1].set_title("Adaptive Threshold")
    axes[2].imshow(rotated, cmap='gray')
    axes[2].axis('off')
    axes[2].set_title("Rotated Adaptive")
    plt.show()


def task8_masking(image_path1="image1.jpg", image_path2="image2.jpg"):
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    if img1 is None or img2 is None:
        print("Error: Image not found.")
        return
    size = (500, 500)
    img1 = cv2.resize(img1, size)
    img2 = cv2.resize(img2, size)
    mask = np.zeros(size, dtype=np.uint8)
    cv2.circle(mask, (250, 250), 200, 255, thickness=-1)
    fg = cv2.bitwise_and(img1, img1, mask=mask)
    inverse_mask = cv2.bitwise_not(mask)
    bg = cv2.bitwise_and(img2, img2, mask=inverse_mask)
    result = cv2.bitwise_or(fg, bg)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(result_rgb)
    plt.show()


def task9_pixel_profiling(image_path="image.jpg"):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    red = image_rgb[:, :, 0].flatten()
    green = image_rgb[:, :, 1].flatten()
    blue = image_rgb[:, :, 2].flatten()
    df = pd.DataFrame({"Red": red, "Green": green, "Blue": blue})
    print(df.describe())


if __name__ == "__main__":
    validate_environment()
    run_grocery_manager_demo()
    run_student_records_demo()
    task3_load_and_show()
    task4_geometry_canvas()
    task5_blur_roi()
    task6_alpha_blend_caption()
    task7_rotation_thresholding()
    task8_masking()
    task9_pixel_profiling()
