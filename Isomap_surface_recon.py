import os
from matplotlib import pyplot as plt
import nrrd
import numpy as np
import argparse
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from sklearn.decomposition import PCA
import pyvista as pv
import open3d as o3d 
from midline_helper_simplified import *
from scipy.ndimage import label as connected_components
from sklearn.manifold import Isomap


def get_rotation_matrix(normal, pca):
    # First rotation: Align the normal with [0, 0, -1]
    target_normal = np.array([0, 0, -1])
    v = np.cross(normal, target_normal)
    c = np.dot(normal, target_normal)
    if np.linalg.norm(v) != 0:
        skew_symmetric = np.array([[0, -v[2], v[1]],
                                [v[2], 0, -v[0]],
                                [-v[1], v[0], 0]])
        init_rotation_matrix = (np.eye(3) + skew_symmetric +
                        skew_symmetric @ skew_symmetric * ((1 - c) / (np.linalg.norm(v) ** 2)))
    else:
        init_rotation_matrix = np.eye(3) if c > 0 else -np.eye(3)

    inverse_rotation_matrix = np.linalg.inv(init_rotation_matrix)
    return init_rotation_matrix, inverse_rotation_matrix

    # Rotate PCA components using the first rotation matrix
    rotated_components = init_rotation_matrix @ pca.components_.T

    # Second rotation: Align the second component with [0, 1, 0] in the xy-plane
    # Project the second component onto the xy-plane
    second_component_xy = rotated_components[:2, 1]
    # Desired forward direction in the xy-plane
    forward_xy = np.array([0, 1])

    # Calculate the angle between the projected second component and the forward direction
    angle = np.arctan2(second_component_xy[1], second_component_xy[0]) - np.arctan2(forward_xy[1], forward_xy[0])

    # Create a rotation matrix around the z-axis
    cos_angle = np.cos(-angle)
    sin_angle = np.sin(-angle)
    rotation_z = np.array([[cos_angle, -sin_angle, 0],
                        [sin_angle, cos_angle, 0],
                        [0, 0, 1]])

    # Combine the rotations
    rotation_matrix = rotation_z @ init_rotation_matrix
    inverse_rotation_matrix = np.linalg.inv(rotation_matrix)
    return rotation_matrix, inverse_rotation_matrix

def visualize_registration_step(**kwargs):
    iteration = kwargs['iteration']
    error = kwargs['error']
    X = kwargs['X']  # target points
    Y = kwargs['Y']  # current source points
    
    if iteration % 25 == 0:  # Visualize every 10 iterations to avoid slowdown
        # Create point clouds for visualization
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(X)
        target_pcd.paint_uniform_color([1, 0, 0])  # Red for target
        
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(Y)
        source_pcd.paint_uniform_color([0, 1, 0])  # Green for source
        
        print(f'Iteration: {iteration}, Error: {error}')
        o3d.visualization.draw_geometries([target_pcd, source_pcd],
                                        zoom=0.8,
                                        front=[0, -0.5, 0],  # Up and to the side
                                        lookat=[0, 0, 0],
                                        up=[0, 0, 1],
                                        window_name=f'Iteration {iteration}')

def preprocess_point_cloud(label_data, ds_factor=4, normals=False):
    """
    Creates and preprocesses point cloud from label data including PCA, rotation and translation.
    
    Args:
        label_data: Binary mask of the region of interest
        ds_factor: Downsampling factor for voxelization
        
    Returns:
        front_pcd: Processed front-facing point cloud
        bbox_center: Original bounding box center for reverse transformation
        rotation_matrix: Applied rotation matrix
        inverse_rotation_matrix: Inverse rotation matrix for reverse transformation
    """
    # Create initial point cloud from coordinates
    coords = np.array(np.where(label_data)).T
    mediod = np.median(coords, axis=0)
    
    # Get rotation matrices from PCA
    pca = PCA(n_components=3)
    pca.fit(coords)
    normal = pca.components_[-1]
    rotation_matrix, inverse_rotation_matrix = get_rotation_matrix(normal, pca)

    # Create and rotate point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.rotate(rotation_matrix, center=mediod)

    # Center the point cloud
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox_center = bbox.get_center()
    pcd.translate(-bbox_center)

    # Estimate and orient normals
    if normals: 
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=50))
        pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0., 0., 1.]))

    # Get front-facing points using hidden point removal
    front_camera = [0,0,135]
    diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
    _, front_pcd_map = pcd.hidden_point_removal(front_camera, radius=50*diameter)
    front_pcd = pcd.select_by_index(front_pcd_map)
    
    # Downsample and recompute normals
    front_pcd = front_pcd.voxel_down_sample(ds_factor)
    front_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=8, max_nn=30))
    front_pcd.orient_normals_towards_camera_location(camera_location=front_camera)
    
    return front_pcd, bbox_center, mediod, inverse_rotation_matrix

def isomap_mesh_fitting(label_data, file_path, label, ds_factor=4, vis=False, plot=False, slices=50, alpha_factor=4, verbose=False):
    front_pcd, bbox_center, mediod, inverse_rotation_matrix = preprocess_point_cloud(label_data, ds_factor)
    if verbose:
        print(f"created front pcd with normals and num points: {len(front_pcd.points)}")
    
    target_landmark_indices = get_edge_points_isomap(front_pcd, slices=slices, plot=plot)

    self_intersection = True
    target_points = len(front_pcd.points)//4
    i = 0
    alpha = ds_factor * alpha_factor #maximum delauney 2d edge length

    # Iteratively reduce point cloud and mesh size until no self intersections or max iterations
    while self_intersection:
        isomap_mesh = generate_mesh_with_isomap(front_pcd, target_points=target_points, plot=False, landmark_indices=target_landmark_indices, alpha=alpha)
        isomap_mesh = pyvista_to_o3d(isomap_mesh)
        self_intersection = isomap_mesh.is_self_intersecting()
        if verbose:
            print(f"self intersection iter {i}: {self_intersection}")
        if not self_intersection:
            break
        
        target_points = target_points//2
        slices = slices//2
        alpha *= 1.5
        target_landmark_indices = get_edge_points_isomap(front_pcd, slices=slices, plot=False)
        i += 1
        if i >= 5:
            break

    # o3d_isomap_mesh = pyvista_to_o3d(isomap_mesh)
    # isomap_mesh.paint_uniform_color([0, 0, 1])
    if verbose:
        print_mesh_info(isomap_mesh, "isomap")

    if vis: 
        o3d.visualization.draw_geometries([front_pcd, isomap_mesh],
                                        zoom=0.8,
                                    front=[0, 0, -1],
                                    lookat=[0, 0, 0],
                                    up=[0, 1, 0],
                                    mesh_show_back_face=True)

    # After visualization, reverse the transformation
    isomap_mesh.translate(bbox_center)
    isomap_mesh.rotate(inverse_rotation_matrix, center=mediod)

    z,y,x = [int(coord) for coord in file_path.split('/')[-2].split('_')[:3]]
    isomap_mesh.translate([z,y,x])


    
    # Save the mesh
    base_name = os.path.basename(file_path).replace('.nrrd', f'_label{label}.ply')
    output_dir = os.path.join(os.path.dirname(file_path), 'isomap_mesh')
    os.makedirs(output_dir, exist_ok=True)
    o3d.io.write_triangle_mesh(os.path.join(output_dir, base_name), isomap_mesh)
    
    return isomap_mesh

def process_file(file_path, vis=False, min_threshold=100000, ds_factor=4, plot=False, slices=50, alpha_factor=4, verbose=False):
    data, _ = nrrd.read(file_path)
    label_values = np.unique(data)
    label_values = label_values[label_values != 0]  # Exclude background

    meshes = []
    voxel_counts = []  # List to store voxel counts for each component

    for label in label_values:
        label_data = data == label
        labeled_array, num_features = connected_components(label_data)

        for component in range(1, num_features + 1):
            component_mask = labeled_array == component
            voxel_count = np.sum(component_mask)
            if voxel_count <= min_threshold:
                continue
            voxel_counts.append(voxel_count)  # Add voxel count to the list
            mesh = isomap_mesh_fitting(component_mask, file_path, label, 
                                               ds_factor=ds_factor, 
                                               vis=vis,
                                               plot=plot,
                                               slices=slices,
                                               alpha_factor=alpha_factor,
                                               verbose=verbose) 
            if mesh is not None:
                meshes.append(mesh)

    return meshes, voxel_counts

def process_file_wrapper(args):
    file_path, vis, min_threshold, ds_factor, plot, slices, alpha_factor, verbose = args
    process_file(file_path, vis, min_threshold, ds_factor, plot, slices, alpha_factor, verbose)
    return

def process_nrrd_files(directory_path, test_cubes=None, suffix=None, vis=False, n_std=1, 
                      min_threshold=10000, ds_factor=4, plot=False, slices=50, alpha_factor=4, verbose=False):
    nrrd_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if suffix:
                if file.endswith(suffix):
                    file_path = os.path.join(root, file)
                    if test_cubes:
                        if any(cube in file_path for cube in test_cubes):
                            nrrd_files.append(file_path)
                    else:
                        nrrd_files.append(file_path)
            else:
                if file.endswith('_mask.nrrd'):
                    file_path = os.path.join(root, file)
                    if test_cubes:
                        if any(cube in file_path for cube in test_cubes):
                            nrrd_files.append(file_path)
                    else:
                        nrrd_files.append(file_path)
    
    with ProcessPoolExecutor() as executor:
        file_args = [(f, vis, min_threshold, ds_factor, plot, slices, alpha_factor, verbose) for f in nrrd_files]
        list(tqdm(executor.map(process_file_wrapper, file_args), 
                            total=len(nrrd_files), desc="Processing files"))

def get_edge_points_isomap(pcd, slices = 50, n_neighbors=5, plot=False):
    """Reduce point cloud dimensionality with Isomap and find edge landmark points.
    
    Args:
        pcd: open3d PointCloud
        n_neighbors: Number of neighbors for Isomap (default 5)
        plot: Whether to show Isomap embedding plot
        
    Returns:
        landmark_indices: Indices of edge landmark points in original point cloud
    """
    # Convert point cloud to numpy array
    points = np.asarray(pcd.points)
    
    # Create and fit Isomap
    isomap = Isomap(n_neighbors=n_neighbors, n_components=2, metric='euclidean')
    embedding = isomap.fit_transform(points)
    
    # Find center of embedding
    center = np.mean(embedding, axis=0)
    
    num_slices = slices  # Number of radial slices

    # Calculate angles of each point relative to the center
    delta_x = embedding[:, 0] - center[0]
    delta_y = embedding[:, 1] - center[1]
    angles = np.arctan2(delta_y, delta_x)
    angles = (angles + 2 * np.pi) % (2 * np.pi)  # Normalize to [0, 2π)

    # Define slice boundaries
    slice_edges = np.linspace(0, 2 * np.pi, num_slices + 1)

    landmark_indices = []

    for i in range(num_slices):
        start_angle = slice_edges[i]
        end_angle = slice_edges[i + 1]

        # Find points within the current slice
        in_slice = (angles >= start_angle) & (angles < end_angle)

        if np.any(in_slice):
            # Compute distances from center for points in the slice
            distances = np.linalg.norm(embedding[in_slice] - center, axis=1)
            # Index of the furthest point within the slice
            furthest_idx_in_slice = np.argmax(distances)
            # Original index in the point cloud
            original_idx = np.where(in_slice)[0][furthest_idx_in_slice]
            landmark_indices.append(original_idx)

    # Sort landmarks clockwise starting from -π radians
    landmark_angles = angles[landmark_indices]
    # Adjust angles to start from -π
    adjusted_angles = (landmark_angles - (-np.pi)) % (2 * np.pi)
    sorted_order = np.argsort(adjusted_angles)
    landmark_indices = [landmark_indices[i] for i in sorted_order]

    if plot:
        plt.figure(figsize=(10,10))
        plt.scatter(embedding[:, 0], embedding[:, 1], s=1, alpha=0.5, label='All points')
        plt.scatter(center[0], center[1], c='red', s=100, marker='*', label='Center')
        
        # Plot edge landmarks
        landmark_embedding = embedding[landmark_indices] 
        plt.scatter(landmark_embedding[:, 0], landmark_embedding[:, 1], c='orange', s=100, marker='o', label='Edge landmarks')
        
        # Visualize slice regions
        radius = np.max(np.linalg.norm(embedding - center, axis=1))
        for angle in slice_edges:
            x = [center[0], center[0] + radius * np.cos(angle)]
            y = [center[1], center[1] + radius * np.sin(angle)]
            plt.plot(x, y, 'k--', alpha=0.3)
        
        plt.title('Isomap embedding with edge landmarks and slice regions')
        plt.legend()
        plt.show()

    return landmark_indices

def get_landmark_points_isomap(pcd, n_neighbors=5, plot=False):
    """Reduce point cloud dimensionality with Isomap and find corner landmark points.
    
    Args:
        pcd: open3d PointCloud
        n_neighbors: Number of neighbors for Isomap (default 5)
        plot: Whether to show Isomap embedding plot
        
    Returns:
        landmark_indices: Indices of corner landmark points in original point cloud
    """
    # Convert point cloud to numpy array
    points = np.asarray(pcd.points)
    
    # Create and fit Isomap
    isomap = Isomap(n_neighbors=n_neighbors, n_components=2, metric='euclidean')
    embedding = isomap.fit_transform(points)
    
    if plot:
        plt.figure(figsize=(10,10))
        plt.scatter(embedding[:, 0], embedding[:, 1], s=1)
        plt.title('Isomap embedding of point cloud')
        plt.show()
    
    # Find center of embedding
    center = np.mean(embedding, axis=0)
    
    # Split into quadrants based on center
    q2_mask = (embedding[:,0] >= center[0]) & (embedding[:,1] >= center[1])  # Upper right
    q1_mask = (embedding[:,0] < center[0]) & (embedding[:,1] >= center[1])   # Upper left
    q3_mask = (embedding[:,0] < center[0]) & (embedding[:,1] < center[1])    # Lower left
    q4_mask = (embedding[:,0] >= center[0]) & (embedding[:,1] < center[1])   # Lower right

    # Find furthest point in each quadrant
    corner_indices = []
    corner_points = []
    for mask in [q1_mask, q2_mask, q3_mask, q4_mask]:
        if np.any(mask):
            quadrant_points = embedding[mask]
            distances = np.linalg.norm(quadrant_points - center, axis=1)
            furthest_idx = np.where(mask)[0][np.argmax(distances)]
            corner_indices.append(furthest_idx)
            corner_points.append(points[furthest_idx])
    
    # Find point closest to center
    distances_to_center = np.linalg.norm(embedding - center, axis=1)
    center_idx = np.argmin(distances_to_center)
    
    # Sort corners by their spatial relationship to match source mesh corners
    corner_points = np.array(corner_points)
    center_point = points[center_idx]
    
    # Calculate angles from center in XY plane
    vectors_to_center = corner_points - center_point
    angles = np.arctan2(vectors_to_center[:, 1], vectors_to_center[:, 0])
    
    # Sort indices by angle to get consistent ordering (counterclockwise from -π)
    sorted_order = np.argsort(angles)
    corner_indices = np.array(corner_indices)[sorted_order]
    
    # Add center point at the end
    landmark_indices = np.append(corner_indices, center_idx)
    
    # Visualize landmarks on original point cloud
    if plot:
        landmark_spheres = []
        colors = [[1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1]]  # Different color for each corner
        for idx, point in enumerate(points[landmark_indices]):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=3.0)
            sphere.translate(point)
            sphere.paint_uniform_color(colors[idx])
            landmark_spheres.append(sphere)
        
        pcd.paint_uniform_color([0, 1, 0])  # Green for original points
        o3d.visualization.draw_geometries([pcd] + landmark_spheres)
    
    return landmark_indices

def visualize_landmarks_and_meshes(planar_mesh, front_pcd, source_landmark_indices, target_landmark_indices):
    # Define 5 distinct colors for landmarks
    colors = [
        [1, 0, 0],   # Red
        [0, 1, 0],   # Green
        [0, 0, 1],   # Blue
        [1, 1, 0],   # Yellow
        [1, 0, 1]    # Magenta
    ]
    
    # Create visualization elements
    vis_objects = []
    
    # Add planar mesh
    mesh_vis = o3d.geometry.TriangleMesh(planar_mesh)
    mesh_vis.paint_uniform_color([0.5,0.5,0.5]) 
    vis_objects.append(mesh_vis)
    
    # Add front point cloud
    pcd_vis = o3d.geometry.PointCloud(front_pcd)
    pcd_vis.paint_uniform_color([1,0,0]) 
    vis_objects.append(pcd_vis)
    
    # Add landmark spheres for planar mesh
    mesh_points = np.asarray(planar_mesh.vertices)
    for idx, landmark_idx in enumerate(source_landmark_indices):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=4.0)
        sphere.translate(mesh_points[landmark_idx])
        sphere.paint_uniform_color([0,1,0])
        vis_objects.append(sphere)
    
    # Add landmark spheres for point cloud
    pcd_points = np.asarray(front_pcd.points)
    for idx, landmark_idx in enumerate(target_landmark_indices):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=4.0)
        sphere.translate(pcd_points[landmark_idx])
        sphere.paint_uniform_color([1,0,0])
        vis_objects.append(sphere)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50, origin=[-128, -128, -128])
    vis_objects.append(coord_frame)
    
    # Visualize everything
    o3d.visualization.draw_geometries(vis_objects,
                                    zoom=0.8,
                                    front=[0, 0, -1],
                                    lookat=[0, 0, 0],
                                    up=[0, 1, 0],
                                    mesh_show_back_face=True,
                                    mesh_show_wireframe=False)

def get_edge_points_mesh(mesh, slices=50, plot=False):
    """Find edge points of a 2D planar mesh using radial slicing.
    
    Args:
        mesh: open3d TriangleMesh (assumed to be planar)
        slices: Number of radial slices (default 50)
        plot: Whether to show visualization
        
    Returns:
        landmark_indices: Indices of edge points sorted clockwise from -π
    """
    # Convert vertices to numpy array and project to 2D (assuming XY plane)
    points = np.asarray(mesh.vertices)
    points_2d = points[:, :2]  # Take only X and Y coordinates
    
    # Find center of mesh
    center = np.mean(points_2d, axis=0)
    
    # Calculate angles of each point relative to the center
    delta_x = points_2d[:, 0] - center[0]
    delta_y = points_2d[:, 1] - center[1]
    angles = np.arctan2(delta_y, delta_x)
    angles = (angles + 2 * np.pi) % (2 * np.pi)  # Normalize to [0, 2π)

    # Define slice boundaries
    slice_edges = np.linspace(0, 2 * np.pi, slices + 1)
    
    landmark_indices = []
    
    # Find furthest point in each slice
    for i in range(slices):
        start_angle = slice_edges[i]
        end_angle = slice_edges[i + 1]
        
        # Find points within current slice
        in_slice = (angles >= start_angle) & (angles < end_angle)
        
        if np.any(in_slice):
            # Compute distances from center for points in slice
            distances = np.linalg.norm(points_2d[in_slice] - center, axis=1)
            # Get index of furthest point
            furthest_idx_in_slice = np.argmax(distances)
            original_idx = np.where(in_slice)[0][furthest_idx_in_slice]
            landmark_indices.append(original_idx)
    
    # Sort landmarks clockwise starting from -π
    landmark_angles = angles[landmark_indices]
    adjusted_angles = (landmark_angles - (-np.pi)) % (2 * np.pi)
    sorted_order = np.argsort(adjusted_angles)
    landmark_indices = [landmark_indices[i] for i in sorted_order]
    
    if plot:
        plt.figure(figsize=(10,10))
        plt.scatter(points_2d[:, 0], points_2d[:, 1], s=1, alpha=0.5, label='All points')
        plt.scatter(center[0], center[1], c='red', s=100, marker='*', label='Center')
        
        # Plot edge landmarks
        landmark_points = points_2d[landmark_indices]
        plt.scatter(landmark_points[:, 0], landmark_points[:, 1], 
                   c='orange', s=100, marker='o', label='Edge landmarks')
        
        # Visualize slice regions
        radius = np.max(np.linalg.norm(points_2d - center, axis=1))
        for angle in slice_edges:
            x = [center[0], center[0] + radius * np.cos(angle)]
            y = [center[1], center[1] + radius * np.sin(angle)]
            plt.plot(x, y, 'k--', alpha=0.3)
        
        plt.title('Mesh points with edge landmarks and slice regions')
        plt.legend()
        plt.show()
    
    return landmark_indices

if __name__ == "__main__":
    default_path = "/Users/jamesdarby/Desktop/manually_labelled_cubes/public_s1-8um"
    parser = argparse.ArgumentParser(description="Perform statistical shape modelling on NRRD files in a directory.")
    parser.add_argument("--input_path", type=str, help="Path to the directory containing NRRD files")
    parser.add_argument("--test", action="store_true", help="Run in test mode (process only specific test cubes)")
    parser.add_argument("--suffix", type=str, help="Suffix to match (e.g., '_mask.nrrd')")
    parser.add_argument("--vis", action="store_true", help="Visualize the point clouds, for debugging")
    parser.add_argument("--n_std", type=float, default=1.0, help="Number of standard deviations below the mean voxel count for thresholding")
    parser.add_argument("--min_threshold", type=int, default=100000, help="Minimum voxel count for size thresholding, and exclusion from mean and std calculation")
    parser.add_argument("--target_point_count", type=int, default=None, help="Target number of points for point cloud downsampling")
    parser.add_argument("--ds_factor", type=int, default=4, 
                       help="Downsampling factor for point cloud voxelization")
    parser.add_argument("--plot", action="store_true", 
                       help="Plot visualizations during landmark detection")
    parser.add_argument("--slices", type=int, default=50,
                       help="Number of radial slices for landmark detection on 2d isomap embedding")
    parser.add_argument("--alpha_factor", type=float, default=4,
                       help="Factor to multiply ds_factor by for maximum delaunay edge length")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress information")
    args = parser.parse_args()

    test_cubes = None
    if args.test:
        # test_cubes = ['01744_02256_03792', '01744_02256_04048', '01744_02256_04304', '01744_02000_04048', '01744_02000_04304']
        test_cubes = ['01744_02256_04048']

    # Use the command line argument if provided, otherwise use the default path
    directory_path = args.input_path if args.input_path else default_path
    process_nrrd_files(directory_path, test_cubes, args.suffix, args.vis, args.n_std, 
                                               args.min_threshold, args.ds_factor,
                                                args.plot, args.slices, args.alpha_factor, args.verbose)  # Add args.slices
