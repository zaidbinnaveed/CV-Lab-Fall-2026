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
