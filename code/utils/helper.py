import numpy as np
from scipy.linalg import expm

class Helper:
    @staticmethod
    def lidar_to_cartesian(ranges, angles):
        indices = (ranges > 0.1) & (ranges < 30.0) & np.isfinite(ranges)
        ranges = ranges[indices]
        angles = angles[indices]

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        return np.column_stack((x, y)) 

    @staticmethod
    def exp_map_so3(tau, omega_hat):
        """
        Computes the SO(3) exponential map.

        Args:
            tau (float): Scalar time parameter.
            omega_hat (np.ndarray): 3x3 skew-symmetric matrix.

        Returns:
            np.ndarray: 3x3 rotation matrix.
        """
        return expm(tau * omega_hat)

    @staticmethod
    def exp_map_se3(tau, twist):
        omega_hat = twist[:3, :3]
        v = twist[:3, 3]

    # Compute rotation matrix
        theta = np.linalg.norm([omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]])
        if np.isclose(theta, 0):
            R = np.eye(3)
        else:
            R = expm(tau * omega_hat)

    # Compute translation
        if np.isclose(theta, 0):
            t = tau * v
        else:
            A = (np.sin(theta * tau)) / theta
            B = (1 - np.cos(theta * tau)) / (theta**2)
            t = (tau * np.eye(3) + B * omega_hat + (tau - A) / (theta**2) * (omega_hat @ omega_hat)) @ v

    # Construct transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T 

    @staticmethod
    def rotation_angle_from_matrix(R):
    # Ensure the input is a valid rotation matrix
        assert R.shape == (3, 3), "Input must be a 3x3 matrix."
        assert np.allclose(R @ R.T, np.eye(3)), "Input must be a valid rotation matrix."

    # Compute trace and clip to valid range
        trace_value = (np.trace(R) - 1) / 2
        trace_value = np.clip(trace_value, -1.0, 1.0)

    # Handle small angles using Taylor series approximation
        if np.isclose(trace_value, 1.0):
            theta = np.sqrt(2 * (1 - trace_value))  # Taylor series for small angles
        else:
            theta = np.arccos(trace_value)

        return theta
