import numpy as np
import matplotlib.pyplot as plt
import gtsam
from .timer import Timer

class Tester:
    @staticmethod
    def bresenham2D(sx, sy, ex, ey):
        """Bresenham's ray tracing algorithm in 2D."""
        sx, sy, ex, ey = map(round, [sx, sy, ex, ey])
        dx, dy = abs(ex - sx), abs(ey - sy)
        steep = dy > dx
        if steep:
            dx, dy = dy, dx
        
        q = np.zeros((dx + 1,)) if dy == 0 else np.append(0, np.greater_equal(
            np.diff(np.mod(np.arange(np.floor(dx / 2), -dy * dx + np.floor(dx / 2) - 1, -dy), dx)), 0))
        
        if steep:
            y = np.arange(sy, ey + 1) if sy <= ey else np.arange(sy, ey - 1, -1)
            x = sx + np.cumsum(q) if sx <= ex else sx - np.cumsum(q)
        else:
            x = np.arange(sx, ex + 1) if sx <= ex else np.arange(sx, ex - 1, -1)
            y = sy + np.cumsum(q) if sy <= ey else sy - np.cumsum(q)
        
        return np.vstack((x, y))

    def test_bresenham2D(self):
        print("Testing bresenham2D...")
        r1 = self.bresenham2D(0, 1, 10, 5)
        r1_ex = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5]])
        r2 = self.bresenham2D(0, 1, 9, 6)
        r2_ex = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 2, 3, 3, 4, 4, 5, 5, 6]])
        
        if np.all(r1 == r1_ex) and np.all(r2 == r2_ex):
            print("...Test passed.")
        else:
            print("...Test failed.")
        
        start_time = time.time()
        for _ in range(1000):
            self.bresenham2D(0, 1, 500, 200)
        print(f"1000 raytraces: --- {time.time() - start_time} seconds ---") 

    @staticmethod
    def test_create_pos2():
        # Create 2D pose with x, y, and theta (rotation)
        pose = gtsam.Pose2(1.0, 2.0, 0.5)
        print("Pose2 created:", pose)
        return pose

    @staticmethod
    def test_create_prior():
        # Create prior factor on a Pose2
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas([0.1, 0.1, 0.1])
        pose_key = gtsam.symbol('x', 1)
        prior_factor = gtsam.PriorFactorPose2(pose_key, gtsam.Pose2(0, 0, 0), prior_noise)
        print("Prior factor created:", prior_factor)
        return prior_factor
