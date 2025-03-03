import matplotlib.pyplot as plt
import os
import open3d as o3d
import numpy as np

class Plotter:
    def __init__(self):
        plt.ion()
   
    @staticmethod
    def view_trajectory(positions, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True) 
        plt.plot(positions[:, 0, 3], positions[:, 1, 3], marker='o', linestyle='-', markersize=1)
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title("2D Trajectory (X-Y Plane) - Exact Integration")
        plt.grid()
        plt.savefig(output_path)

    @staticmethod
    def compare_trajectory(positions, optimized_positions, output_path):
        plt.figure(figsize=(10, 8))
        plt.plot(positions[:, 0, 3], positions[:, 1, 3], color='blue', label='Original Model', linestyle='-', marker='o', alpha=0.6)
        plt.plot(optimized_positions[:, 0, 3], optimized_positions[:, 1, 3], color='red', label='Optimized Model', linestyle='-', marker='x')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Original and Optimized Trajectories')
        plt.legend()
        plt.savefig(output_path)
        plt.show()

    @staticmethod
    def view_icp(source_pc, target_pc, pose, output_path):
        '''
        Save the result of ICP as an image
        source_pc: numpy array, (N, 3)
        target_pc: numpy array, (N, 3)
        pose: SE(4) numpy array, (4, 4)
        save_path: str, path to save the image
        '''
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source_pc.reshape(-1, 3))
        source_pcd.paint_uniform_color([0, 0, 1])   ## BLUE 
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_pc.reshape(-1, 3))
        target_pcd.paint_uniform_color([1, 0, 0]) ## RED
        target_pcd.transform(pose)
        #source_pcd.transform(pose.T)
    
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1280, height=720)    
        vis.add_geometry(source_pcd)
        vis.add_geometry(target_pcd)
    
        view_control = vis.get_view_control()
        view_control.set_lookat([0, 0, 0])
        view_control.set_up([0, 0, 1])
        view_control.set_front([-1, 1, 0])
        view_control.set_zoom(0.8)    
 
        vis.update_geometry(source_pcd)
        vis.update_geometry(target_pcd)
        vis.poll_events()
        vis.update_renderer()
    
        vis.capture_screen_image(output_path)
        vis.destroy_window()

    @staticmethod
    def multiview(source_pc, target_pc, pose, output_path):
        os.makedirs(output_path+"/views", exist_ok=True)
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source_pc.reshape(-1, 3))
        source_pcd.paint_uniform_color([0, 0, 1])  # Blue color for source points
    
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_pc.reshape(-1, 3))
        target_pcd.paint_uniform_color([1, 0, 0])  # Red color for target points
    
        target_pcd.transform(pose)
    
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1280, height=720)
    
        vis.add_geometry(source_pcd)
        vis.add_geometry(target_pcd)
    
        view_control = vis.get_view_control()
        view_control.set_lookat([0, 0, 0])
        view_control.set_zoom(0.8)

        view_control.set_front([0, 0, 1])
        view_control.set_up([0, 1, 0])     
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(os.path.join(output_path, "views/top_view.jpg"))

        view_control.set_front([-1, 1, 0])
        view_control.set_up([0, 0, 1])
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(os.path.join(output_path, "views/left_side_view.jpg"))
        
        view_control.set_front([-1, 1, 1])
        view_control.set_up([0, 0, 1])
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(os.path.join(output_path, "views/left_45_view.jpg"))

        view_control.set_front([1, -1, 1]) 
        view_control.set_up([0, 0, 1])
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(os.path.join(output_path, "views/right_45_view.jpg"))

        view_control.set_front([1, -1, 0])
        view_control.set_up([0, 0, 1])
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(os.path.join(output_path, "views/right_side_view.png"))

        view_control.set_front([0, 0, -1])
        view_control.set_up([0, -1, 0])
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(os.path.join(output_path, "views/bottom_view.png"))
        vis.destroy_window()

    @staticmethod
    def plot_map(mapdata, cmap="binary"):
        plt.imshow(mapdata.T, origin="lower", cmap=cmap)
    
    @staticmethod
    def generate_map(filename="../data/test_ranges.npy"):
        MAP = {
            'res': np.array([0.05, 0.05]),
            'min': np.array([-20.0, -20.0]),
            'max': np.array([20.0, 20.0]),
        }
        MAP['size'] = np.ceil((MAP['max'] - MAP['min']) / MAP['res']).astype(int)
        MAP['size'][MAP['size'] % 2 == 0] += 1  # Ensure odd size for centering
        MAP['map'] = np.zeros(MAP['size'])
        
        ranges = np.load(filename)
        angles = np.radians(np.arange(-135, 135.25, 0.25))
        valid1 = (ranges < 30) & (ranges > 0.1)
        points = np.column_stack((ranges * np.cos(angles), ranges * np.sin(angles)))
        cells = np.floor((points - MAP['min']) / MAP['res']).astype(int)
        valid2 = np.all((cells >= 0) & (cells < MAP['size']), axis=1)
        MAP['map'][tuple(cells[valid1 & valid2].T)] = 1
        
        return MAP

    def test_map(self):
        MAP = self.generate_map()
        plt.figure()
        plt.plot(MAP['map'], '.k')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Lidar scan')
        plt.axis('equal')
        plt.figure()
        self.plot_map(MAP['map'], cmap='binary')
        plt.title('Grid map')
        plt.show()
    

