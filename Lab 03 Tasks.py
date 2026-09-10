"""
Lab 03 Tasks - Geometric Image Transformations
================================================
Implements Tasks 1-10 exactly as specified in Lab_03_Tasks.pdf.

Convention used throughout:
    - Image coordinates are (x, y) with x = column, y = row.
    - Points are represented as column vectors [x, y] or homogeneous [x, y, 1].
    - Matrices are built BY HAND (manual entry of sin/cos/scale/shear terms,
      or by solving linear systems) as each task requires -- OpenCV's
      warpAffine / warpPerspective are only used to *apply* the matrix we
      already built, never to build it for us (except where a task
      explicitly allows solving via matching points, e.g. Task 8/9).

Run this file directly to generate a synthetic test image per task and
save the "before/after" results into ./outputs/.
"""

import numpy as np
import cv2
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def save(name, img):
    path = os.path.join(OUT_DIR, name)
    cv2.imwrite(path, img)
    print(f"saved -> {path}")


# ---------------------------------------------------------------------------
# TASK 1: The Tiny Fingerprint (Linear - Scale)
# ---------------------------------------------------------------------------
def task1_scale_fingerprint(img, sx=3.0, sy=3.0):
    """
    Manually build a 2x2 scaling matrix and enlarge the image by 300% on
    both axes (scale factor = 3), then re-center it so it doesn't just
    grow off the top-left corner.
    """
    h, w = img.shape[:2]

    # 1. Manually build the 2x2 linear scaling matrix
    S = np.array([
        [sx, 0.0],
        [0.0, sy]
    ], dtype=np.float64)

    # 2. A pure 2x2 linear map is anchored at the origin (0,0), so scaling
    #    directly would push the fingerprint off-screen to the bottom-right.
    #    To keep it centered we compute how much bigger the canvas becomes
    #    and shift by half that amount.
    new_w, new_h = int(round(w * sx)), int(round(h * sy))
    dx = (new_w - w) / 2.0
    dy = (new_h - h) / 2.0

    # 3. Embed the 2x2 linear matrix into a 3x3 affine matrix with the
    #    centering translation, then apply it.
    M = np.array([
        [S[0, 0], S[0, 1], dx],
        [S[1, 0], S[1, 1], dy]
    ], dtype=np.float64)

    result = cv2.warpAffine(img, M, (new_w, new_h))
    return result, S, M


# ---------------------------------------------------------------------------
# TASK 2: The Dizzy Satellite (Linear - Rotation)
# ---------------------------------------------------------------------------
def task2_derotate_satellite(img, angle_deg=45.0):
    """
    Formulate a 2x2 rotation matrix from sine/cosine, spin the image back
    to true north (i.e. undo the 45-degree tilt), and manually compute the
    new canvas size so corners aren't clipped.
    """
    h, w = img.shape[:2]

    # We want to UNDO a +45 degree tilt, so we rotate by -angle_deg.
    theta = np.deg2rad(-angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # 1. Manually build the 2x2 rotation matrix
    R = np.array([
        [cos_t, -sin_t],
        [sin_t,  cos_t]
    ], dtype=np.float64)

    # 2. Manually compute the new bounding-box dimensions so the rotated
    #    corners aren't chopped off:
    #    new_w = |w*cos| + |h*sin| ,  new_h = |w*sin| + |h*cos|
    new_w = int(np.ceil(abs(w * cos_t) + abs(h * sin_t)))
    new_h = int(np.ceil(abs(w * sin_t) + abs(h * cos_t)))

    # 3. Rotation about the origin also needs a translation to re-center
    #    the (now larger) canvas, exactly like Task 1.
    cx, cy = w / 2.0, h / 2.0                 # old center
    new_cx, new_cy = new_w / 2.0, new_h / 2.0  # new center

    # tx, ty chosen so that the old center maps to the new center:
    # new_center = R * old_center + t  =>  t = new_center - R * old_center
    old_center = np.array([cx, cy])
    t = np.array([new_cx, new_cy]) - R @ old_center

    M = np.array([
        [R[0, 0], R[0, 1], t[0]],
        [R[1, 0], R[1, 1], t[1]]
    ], dtype=np.float64)

    result = cv2.warpAffine(img, M, (new_w, new_h))
    return result, R, M


# ---------------------------------------------------------------------------
# TASK 3: The Fast Train Barcode (Linear - Shear)
# ---------------------------------------------------------------------------
def task3_deshear_barcode(img, shear_factor=-0.5):
    """
    Construct a 2x2 shear matrix that pushes pixels horizontally to undo a
    rightward slant. A positive `shear_factor` slants right, so to CORRECT
    a rightward slant we apply the negative of that factor.
    """
    h, w = img.shape[:2]

    # 1. Manually build the 2x2 horizontal shear matrix:
    #    x' = x + shear_factor * y
    #    y' = y
    Sh = np.array([
        [1.0, shear_factor],
        [0.0, 1.0]
    ], dtype=np.float64)

    # 2. Shearing horizontally shifts columns outward; widen the canvas so
    #    nothing is clipped, and compute the needed x-offset.
    extra_w = int(np.ceil(abs(shear_factor) * h))
    new_w = w + extra_w
    dx = extra_w if shear_factor < 0 else 0

    M = np.array([
        [Sh[0, 0], Sh[0, 1], dx],
        [Sh[1, 0], Sh[1, 1], 0.0]
    ], dtype=np.float64)

    result = cv2.warpAffine(img, M, (new_w, h))
    return result, Sh, M


# ---------------------------------------------------------------------------
# TASK 4: The Hidden Treasure Map (Affine - Translation)
# ---------------------------------------------------------------------------
def task4_translate_map(img, shift_left=150, shift_up=80):
    """
    Upgrade to a 3x3 homogeneous matrix and build a pure translation matrix
    that slides the map so the hidden 'X' (currently 150px left and 80px
    up of the visible frame) comes back on screen. Bringing something back
    from "off-screen to the left/up" means sliding the whole map
    RIGHT and DOWN by that amount.
    """
    h, w = img.shape[:2]
    tx, ty = shift_left, shift_up  # move right by 150, down by 80

    # 1. Manually build the 3x3 translation matrix
    T = np.array([
        [1.0, 0.0, tx],
        [0.0, 1.0, ty],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # 2. Apply it (warpAffine wants the top 2x3 rows)
    M = T[:2, :]
    new_w, new_h = w + tx, h + ty
    result = cv2.warpAffine(img, M, (new_w, new_h))
    return result, T


# ---------------------------------------------------------------------------
# TASK 5: The Robot's Assembly Line (Rigid Transformation)
# ---------------------------------------------------------------------------
def task5_rigid_align(img, angle_deg=30.0, tx=40, ty=-25):
    """
    A Rigid Transformation = Rotation + Translation only (no scale/shear),
    so shape is perfectly preserved. Combine both into a single 3x3 matrix
    and apply them simultaneously in one warp.
    """
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    R = np.array([
        [cos_t, -sin_t, 0.0],
        [sin_t,  cos_t, 0.0],
        [0.0,    0.0,   1.0]
    ], dtype=np.float64)

    T = np.array([
        [1.0, 0.0, tx],
        [0.0, 1.0, ty],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # Combine: apply rotation first, then translation -> T @ R
    Rigid = T @ R

    h, w = img.shape[:2]
    M = Rigid[:2, :]
    result = cv2.warpAffine(img, M, (w, h))
    return result, Rigid


# ---------------------------------------------------------------------------
# TASK 6: The Architect's Blueprint (Similarity Transformation)
# ---------------------------------------------------------------------------
def task6_similarity_fix(img, scale=1.8, angle_deg=-20.0, tx=60, ty=30):
    """
    A Similarity Transformation = uniform Scale + Rotation + Translation
    (preserves angles, changes size). Combine all three into one 3x3
    matrix and fix the blueprint in a single operation.
    """
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    S = np.array([
        [scale, 0.0,   0.0],
        [0.0,   scale, 0.0],
        [0.0,   0.0,   1.0]
    ], dtype=np.float64)

    R = np.array([
        [cos_t, -sin_t, 0.0],
        [sin_t,  cos_t, 0.0],
        [0.0,    0.0,   1.0]
    ], dtype=np.float64)

    T = np.array([
        [1.0, 0.0, tx],
        [0.0, 1.0, ty],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # Combine: scale, then rotate, then translate -> T @ R @ S
    Similarity = T @ R @ S

    h, w = img.shape[:2]
    new_w, new_h = int(w * scale) + abs(tx) + 50, int(h * scale) + abs(ty) + 50
    M = Similarity[:2, :]
    result = cv2.warpAffine(img, M, (new_w, new_h))
    return result, Similarity


# ---------------------------------------------------------------------------
# TASK 7: The Glitched Image (General Affine Transformation)
# ---------------------------------------------------------------------------
def task7_fix_affine_glitch(img, src_pts, dst_pts):
    """
    Given 3 matching landmark points (before/after the glitch), solve a
    system of linear equations for the 6 unknowns of a general 2x3 affine
    matrix:
        [x']   [a  b  tx] [x]
        [y'] = [c  d  ty] [y]
                          [1]
    We are given dst_pts = A @ src_pts_h, and want the matrix A that
    UNDOES the glitch, i.e. maps the glitched (dst) points back to the
    clean (src) points.
    """
    src_pts = np.array(src_pts, dtype=np.float64)  # 3 clean landmark points
    dst_pts = np.array(dst_pts, dtype=np.float64)  # 3 glitched landmark points

    # We solve for the affine matrix A_fix such that:
    #     src = A_fix @ [dst; 1]
    # Build the 6x6 linear system A_sys * p = b, where p = [a,b,tx,c,d,ty]
    A_sys = np.zeros((6, 6), dtype=np.float64)
    b = np.zeros((6,), dtype=np.float64)

    for i in range(3):
        x, y = dst_pts[i]
        xp, yp = src_pts[i]
        # row for x' = a*x + b*y + tx
        A_sys[2 * i]     = [x, y, 1, 0, 0, 0]
        b[2 * i]         = xp
        # row for y' = c*x + d*y + ty
        A_sys[2 * i + 1] = [0, 0, 0, x, y, 1]
        b[2 * i + 1]     = yp

    p = np.linalg.solve(A_sys, b)
    a, b_, tx, c, d, ty = p

    M = np.array([
        [a, b_, tx],
        [c, d,  ty]
    ], dtype=np.float64)

    h, w = img.shape[:2]
    result = cv2.warpAffine(img, M, (w, h))
    return result, M


# ---------------------------------------------------------------------------
# TASK 8: The Sidewalk Illusion (Projective - Perspective)
# ---------------------------------------------------------------------------
def task8_birds_eye_view(img, corner_pts, out_size=400):
    """
    Given the 4 corners of chalk art that SHOULD form a perfect square,
    build a perspective transformation matrix that warps them into an
    actual square, producing a top-down "bird's-eye" view.

    We solve the 8-unknown homography by hand via the DLT approach (same
    method used in Task 9) rather than calling a built-in "compute
    homography" convenience function.
    """
    src = np.array(corner_pts, dtype=np.float64)  # 4 distorted corners (image)
    dst = np.array([
        [0, 0],
        [out_size - 1, 0],
        [out_size - 1, out_size - 1],
        [0, out_size - 1]
    ], dtype=np.float64)  # target perfect square

    H = compute_homography_DLT(src, dst)  # manual DLT (see Task 9 helper)
    result = cv2.warpPerspective(img, H, (out_size, out_size))
    return result, H


# ---------------------------------------------------------------------------
# TASK 9: The Stadium Panorama (Projective - Homography)
# ---------------------------------------------------------------------------
def compute_homography_DLT(src_pts, dst_pts):
    """
    Manually compute the Homography matrix H (3x3, up to scale) that maps
    src_pts -> dst_pts using the Direct Linear Transform (DLT) with (at
    least) 4 point correspondences. This is the classic "solve via SVD"
    method: build the 2n x 9 matrix A, take the singular vector for the
    smallest singular value, reshape to 3x3.
    """
    src_pts = np.array(src_pts, dtype=np.float64)
    dst_pts = np.array(dst_pts, dtype=np.float64)
    n = src_pts.shape[0]
    assert n >= 4, "Homography needs at least 4 point correspondences"

    A = []
    for i in range(n):
        x, y = src_pts[i]
        xp, yp = dst_pts[i]
        A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])
    A = np.array(A, dtype=np.float64)

    # Solve via SVD: h is the eigenvector for the smallest singular value
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    H = h.reshape(3, 3)
    H = H / H[2, 2]  # normalize so H[2,2] = 1
    return H


def task9_stadium_panorama(img1, img2, pts_img1, pts_img2):
    """
    Given 4 matching points found in both camera images, compute the
    Homography matrix that projects the second camera's view into the
    first camera's coordinate space, then stitch them into one panorama.
    """
    H = compute_homography_DLT(pts_img2, pts_img1)  # img2 -> img1 space

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Determine output canvas size by projecting img2's corners through H
    corners2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], dtype=np.float64)
    corners2_h = np.hstack([corners2, np.ones((4, 1))])
    proj = (H @ corners2_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]

    all_corners = np.vstack([
        proj,
        [[0, 0], [w1, 0], [w1, h1], [0, h1]]
    ])
    x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)

    # Shift everything so nothing is negative
    shift = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ], dtype=np.float64)

    canvas_size = (x_max - x_min, y_max - y_min)

    warped2 = cv2.warpPerspective(img2, shift @ H, canvas_size)
    warped1 = cv2.warpPerspective(img1, shift, canvas_size)

    # Simple stitch: paste img1 on top, then fill remaining area with img2
    panorama = warped2.copy()
    mask1 = np.any(warped1 > 0, axis=-1)
    panorama[mask1] = warped1[mask1]

    return panorama, H


# ---------------------------------------------------------------------------
# TASK 10: The Master Forger (Full Hierarchy Challenge)
# ---------------------------------------------------------------------------
def task10_insert_painting(painting_img, frame_corners, canvas_size):
    """
    Traverse the FULL transformation hierarchy to insert a flat painting
    into an angled picture frame:
        1. Linear Scale     -> shrink the painting
        2. Rigid Transform  -> move it near the frame (rotation + translation)
        3. Projective Warp  -> map its 4 corners exactly onto the angled
                                frame's 4 corners in the final scene
    """
    h, w = painting_img.shape[:2]

    # --- Step 1: Linear Scale (shrink painting) ---
    scale = 0.5
    Scale2x2 = np.array([
        [scale, 0.0],
        [0.0,   scale]
    ], dtype=np.float64)
    new_w, new_h = int(w * scale), int(h * scale)
    M_scale = np.array([
        [Scale2x2[0, 0], Scale2x2[0, 1], 0],
        [Scale2x2[1, 0], Scale2x2[1, 1], 0]
    ], dtype=np.float64)
    scaled = cv2.warpAffine(painting_img, M_scale, (new_w, new_h))

    # --- Step 2: Rigid Transform (rotate slightly + move near frame) ---
    angle_deg = 5.0
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    tx, ty = 20, 15  # nudge toward the frame's approximate position
    Rigid = np.array([
        [cos_t, -sin_t, tx],
        [sin_t,  cos_t, ty],
        [0.0,    0.0,   1.0]
    ], dtype=np.float64)
    M_rigid = Rigid[:2, :]
    positioned = cv2.warpAffine(scaled, M_rigid, (new_w + 60, new_h + 60))

    # --- Step 3: Projective Transform (warp 4 corners onto angled frame) ---
    ph, pw = positioned.shape[:2]
    src_corners = np.array([
        [0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]
    ], dtype=np.float64)
    dst_corners = np.array(frame_corners, dtype=np.float64)  # angled frame's 4 corners

    H = compute_homography_DLT(src_corners, dst_corners)
    warped_painting = cv2.warpPerspective(positioned, H, canvas_size)

    # Build a mask of the warped painting so we can composite it onto the wall
    mask = cv2.warpPerspective(
        np.ones((ph, pw), dtype=np.uint8) * 255, H, canvas_size
    )
    return warped_painting, mask, (Scale2x2, Rigid, H)


# ---------------------------------------------------------------------------
# Demo / self-test harness
# ---------------------------------------------------------------------------
def make_test_image(w=200, h=150, label="TEST"):
    img = np.full((h, w, 3), 30, dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (w - 10, h - 10), (0, 200, 255), 3)
    cv2.putText(img, label, (15, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    # a few landmark dots for tasks that need matching points
    for (x, y) in [(20, 20), (w - 20, 20), (w - 20, h - 20), (20, h - 20)]:
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    return img


if __name__ == "__main__":
    base = make_test_image()
    save("00_original.png", base)

    r1, S, M1 = task1_scale_fingerprint(base)
    save("task1_scaled.png", r1)

    r2, R, M2 = task2_derotate_satellite(base, angle_deg=45)
    save("task2_derotated.png", r2)

    r3, Sh, M3 = task3_deshear_barcode(base, shear_factor=-0.5)
    save("task3_desheared.png", r3)

    r4, T4 = task4_translate_map(base, shift_left=150, shift_up=80)
    save("task4_translated.png", r4)

    r5, Rigid5 = task5_rigid_align(base, angle_deg=30, tx=40, ty=-25)
    save("task5_rigid.png", r5)

    r6, Sim6 = task6_similarity_fix(base, scale=1.8, angle_deg=-20, tx=60, ty=30)
    save("task6_similarity.png", r6)

    h, w = base.shape[:2]
    src7 = [[20, 20], [w - 20, 20], [20, h - 20]]           # clean landmarks
    dst7 = [[35, 40], [w - 10, 30], [50, h - 15]]           # glitched landmarks
    r7, M7 = task7_fix_affine_glitch(base, src7, dst7)
    save("task7_deglitched.png", r7)

    corners8 = [[30, 40], [w - 15, 10], [w - 30, h - 10], [15, h - 30]]
    r8, H8 = task8_birds_eye_view(base, corners8, out_size=300)
    save("task8_birdseye.png", r8)

    base2 = make_test_image(label="CAM2")
    pts_img1 = [[w - 60, 20], [w - 10, 20], [w - 10, h - 20], [w - 60, h - 20]]
    pts_img2 = [[10, 20], [60, 20], [60, h - 20], [10, h - 20]]
    r9, H9 = task9_stadium_panorama(base, base2, pts_img1, pts_img2)
    save("task9_panorama.png", r9)

    painting = make_test_image(w=150, h=100, label="ART")
    canvas = np.full((400, 400, 3), 60, dtype=np.uint8)
    frame_corners = [[80, 60], [320, 30], [340, 300], [60, 320]]  # angled frame
    warped, mask, mats = task10_insert_painting(painting, frame_corners, (400, 400))
    composite = canvas.copy()
    mask_bool = mask > 0
    composite[mask_bool] = warped[mask_bool]
    save("task10_final_composite.png", composite)

    print("\nAll 10 tasks executed successfully. Check ./outputs/")
