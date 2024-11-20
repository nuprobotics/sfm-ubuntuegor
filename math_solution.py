import numpy as np


def triangulation(
        camera_matrix: np.ndarray,
        camera_position1: np.ndarray,
        camera_rotation1: np.ndarray,
        camera_position2: np.ndarray,
        camera_rotation2: np.ndarray,
        image_points1: np.ndarray,
        image_points2: np.ndarray
):
    """
    :param camera_matrix: first and second camera matrix, np.ndarray 3x3
    :param camera_position1: first camera position in world coordinate system, np.ndarray 3x1
    :param camera_rotation1: first camera rotation matrix in world coordinate system, np.ndarray 3x3
    :param camera_position2: second camera position in world coordinate system, np.ndarray 3x1
    :param camera_rotation2: second camera rotation matrix in world coordinate system, np.ndarray 3x3
    :param image_points1: points in the first image, np.ndarray Nx2
    :param image_points2: points in the second image, np.ndarray Nx2
    :return: triangulated points, np.ndarray Nx3
    """

    P1 = camera_matrix @ np.hstack((camera_rotation1.T, -camera_rotation1.T @ camera_position1))
    P2 = camera_matrix @ np.hstack((camera_rotation2.T, -camera_rotation2.T @ camera_position2))

    num_points = image_points1.shape[0]
    triangulated_points = np.zeros((num_points, 3))

    for i in range(num_points):
        x1, y1 = image_points1[i]
        x2, y2 = image_points2[i]

        A = np.array([
            -x1 * P1[2] + P1[0],
            y1 * P1[2] - P1[1],
            -x2 * P2[2] + P2[0],
            y2 * P2[2] - P2[1]
        ])

        _, _, V = np.linalg.svd(A)
        X = V[-1]  # Solution is the last row of V
        X /= X[3]

        triangulated_points[i] = X[:3]

    return triangulated_points
