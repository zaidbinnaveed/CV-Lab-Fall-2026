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
