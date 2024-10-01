import os
from matplotlib import pyplot as plt
import numpy as np
import nrrd
import time
import concurrent.futures
import argparse
from tqdm import tqdm
from scipy.spatial import cKDTree, KDTree
from sklearn.decomposition import PCA
import pyvista as pv
import open3d as o3d

from midline_helper_simplified import *

"""
The goal of this file is to load in harmonised nrrd cube volumatric instance label files
and then for each instance, find the midline sheet/ manifoldof the instance across cubes.

We will do this by representing each instance, using a kdtree to iterate over local neighbourhoods
where the assumption that the instance can be represented by a single voxelised value intersecting the normal
of the surface holds.

the kdtree local neighbourhood search can then also be used to create a mesh via pyvista's 
delauney_2d over the local neighbourhoods.
"""
def visualize_point_clouds(point_clouds):
    plotter = pv.Plotter()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(point_clouds)))

    for i, (label, point_cloud) in enumerate(point_clouds.items()):
        # Create a PyVista point cloud
        pv_point_cloud = pv.PolyData(point_cloud)

        # Add the point cloud to the plotter with a unique color
        plotter.add_mesh(
            pv_point_cloud,
            point_size=10.0,
            render_points_as_spheres=True,
            color=colors[i][:3],  # RGB values
            label=f"Label {label}"
        )

    plotter.add_legend()
    plotter.show_axes()
    plotter.show()

def mls_projection(point_cloud, query_points, search_radius):
    tree = cKDTree(point_cloud)
    projected_points = []

    for qp in query_points:
        # Find neighbors within search radius
        idx = tree.query_ball_point(qp, r=search_radius)
        neighbors = point_cloud[idx]

        # Compute weights
        distances = np.linalg.norm(neighbors - qp, axis=1)
        weights = np.exp(- (distances / search_radius) ** 2)

        # Compute weighted centroid
        weighted_centroid = np.average(neighbors, axis=0, weights=weights)

        # Compute covariance matrix
        centered_neighbors = neighbors - weighted_centroid
        C = np.dot((weights[:, np.newaxis] * centered_neighbors).T, centered_neighbors)

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        normal = eigenvectors[:, 0]  # Smallest eigenvalue

        # Project query point onto local plane
        qp_proj = qp - np.dot((qp - weighted_centroid), normal) * normal
        projected_points.append(qp_proj)

    return np.array(projected_points)

def create_mesh_with_ball_pivoting(point_cloud, radii=[1, 2, 4, 8]):
    # Convert PyVista PolyData to numpy array if necessary
    if isinstance(point_cloud, pv.PolyData):
        points = point_cloud.points
    elif isinstance(point_cloud, np.ndarray):
        points = point_cloud
    else:
        raise ValueError(f"Unexpected input type: {type(point_cloud)}")

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(10)

    # Create mesh using ball pivoting
    radii = o3d.utility.DoubleVector(radii)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)

    # Optional: Clean up the mesh
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    return mesh

def generate_seed_points(coords, radius, tree):
    """
    Generate seed points to cover the structure using a simple greedy algorithm.
    """
    remaining = set(range(len(coords)))
    seed_points = []
    while remaining:
        idx = remaining.pop()
        seed_point = coords[idx]
        seed_points.append(seed_point)
        # Find neighbors within radius
        neighbors = tree.query_ball_point(seed_point, r=radius)
        remaining -= set(neighbors)
    return np.array(seed_points)

def thin_structure(coords, normals, seed_points, radius, tree):
    """
    Thin the structure by projecting points onto planes perpendicular to the normals.
    """
    thinned_points = []
    for i, seed in enumerate(seed_points):
        # Find points within the neighborhood
        idx = tree.query_ball_point(seed, r=radius)
        neighborhood = coords[idx]
        normal = normals[i]

        # Project points onto the plane
        mean_point = neighborhood.mean(axis=0)
        centered = neighborhood - mean_point
        projected = centered - np.dot(centered, normal[:, np.newaxis]) * normal
        # Keep only the mean point (or you can sample)
        thinned_points.append(mean_point)
    return np.array(thinned_points)

def process_single_file_to_point_clouds(paths):
    input_path, output_path = paths
    zyx = os.path.basename(os.path.dirname(input_path)).split('_')
    zyx = [int(coord) for coord in zyx]
    print("Processing cube: ", zyx)
    radius = 20.0 
    point_clouds = {}  # Dictionary to store point clouds for each label

    try:
        # Load the input NRRD file
        data, header = nrrd.read(input_path)
    except Exception as e:
        print(f"Error reading {input_path}: {e}")
        return None

    unique_labels = np.unique(data)
    for label in unique_labels:
        if label == 0:
            continue
        volume = np.where(data == label, label, 0)
        coords = np.array(np.where(volume == label)).T  # Shape (N, 3)
        # Add zyx values to each entry in coords
        zyx_array = np.array(zyx)
        coords = coords + zyx_array
        # Build KD-tree
        tree = KDTree(coords)

        # Generate seed points
        seed_points = generate_seed_points(coords, radius, tree)

        # Compute normals using PCA for each neighborhood
        normals = []
        for seed in seed_points:
            idx = tree.query_ball_point(seed, r=radius)
            neighborhood = coords[idx]
            if len(neighborhood) >= 3:
                pca = PCA(n_components=3)
                pca.fit(neighborhood)
                normal = pca.components_[-1]  # The normal vector
            else:
                normal = np.array([0, 0, 1])  # Default normal if not enough points
            normals.append(normal)
        normals = np.array(normals)

        # Thin the structure
        thinned_points = thin_structure(coords, normals, seed_points, radius, tree)

        
       
        # Store the point cloud for this label
        point_clouds[label] = pv.PolyData(thinned_points)   

    # Visualization with PyVista for all labels
    # if point_clouds:
    #     visualize_point_clouds(point_clouds)

    return point_clouds


        
def kdtree_surface_reconstruction(point_cloud):

    if isinstance(point_cloud, pv.PolyData):
        # Convert PyVista PolyData to numpy array
        point_cloud = point_cloud.points
    elif isinstance(point_cloud, np.ndarray):
        point_cloud = point_cloud
    else:
        raise ValueError(f"Unexpected input type: {type(point_cloud)}")
    
    # Create a KDTree for efficient nearest neighbor search
    kdtree = cKDTree(point_cloud)

    # Define a radius for the neighborhood
    radius = 30.0  # Adjust this value based on your data

    # sparse seed points that cover every point in the point cloud
    # Initialize uncovered point indices
    uncovered_indices = set(range(len(point_cloud)))

    # Initialize seed point indices
    seed_point_indices = []

    # While there are uncovered points
    while uncovered_indices:
        # Select a point from the uncovered set
        current_index = uncovered_indices.pop()
        seed_point_indices.append(current_index)

        # Find all points within the radius of the current point
        indices_within_radius = kdtree.query_ball_point(point_cloud[current_index], r=radius)

        # Remove these points from the uncovered set
        uncovered_indices.difference_update(indices_within_radius)

        print(f"Seed points selected: {len(seed_point_indices)}, Uncovered points remaining: {len(uncovered_indices)}", end='\r')

    # Extract seed points
    # seed_points = point_cloud[seed_point_indices]

    local_meshes = []

    for i, seed_index in enumerate(seed_point_indices):
        # Find neighboring points
        seed_point = point_cloud[seed_index]
        indices = kdtree.query_ball_point(seed_point, r=radius)
        if len(indices) < 3:
            continue
        local_points = point_cloud[indices]

        # Create local PolyData
        local_polydata = pv.PolyData(local_points)

        # Apply delaunay_2d directly
        try:
            local_mesh = local_polydata.delaunay_2d()
        except:
            continue

        # Add the local mesh to the list
        local_meshes.append(local_mesh)

        print(f"Processed {i+1}/{len(seed_point_indices)} neighborhoods", end='\r')

    # Merge all local meshes
    if local_meshes:
        merged_mesh = local_meshes[0]
        for mesh in local_meshes[1:]:
            merged_mesh = merged_mesh.merge(mesh)
        # Clean the merged mesh
        merged_mesh = merged_mesh.clean(tolerance=1e-5)
        # Smooth the mesh using Laplacian smoothing
        # merged_mesh = merged_mesh.smooth(n_iter=1000, relaxation_factor=0.5, feature_angle=170, feature_smoothing=True, boundary_smoothing=True)
        
        merged_mesh.compute_normals(auto_orient_normals=False, inplace=True)
        
        return merged_mesh
    else:
        print("No local meshes were created.")
        return None
            
        
def process_directory(input_directory, replace=False, test_cubes=[], show_time=False):
    files_to_process = []
    for root, _, files in os.walk(input_directory):
        mask_file_suffix = '_relabeled_mask.nrrd'
        mask_file = next((f for f in files if f.endswith(mask_file_suffix)), None)
        if mask_file:
            input_path = os.path.join(root, mask_file)
            output_path = input_path.replace(mask_file_suffix, '_kdtree_mask_thinned.nrrd')
            if os.path.exists(output_path) and not replace:
                continue
            files_to_process.append((input_path, output_path))
    if len(test_cubes) > 0:
        # Filter files_to_process based on test_cubes
        files_to_process = [
            (input_path, output_path) 
            for input_path, output_path in files_to_process 
            if any(cube in input_path for cube in test_cubes)
        ]

    point_clouds = {}

    # Process files in parallel, creating a dictionary of label values and their associated thinned point clouds
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_file_to_point_clouds, args) for args in files_to_process]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(files_to_process), desc="Processing files", unit="file"):
            result = future.result()
            if result:
                for label, point_cloud in result.items():
                    if label in point_clouds:
                        if isinstance(point_clouds[label], list):
                            point_clouds[label].append(point_cloud)
                        else:
                            point_clouds[label] = [point_clouds[label], point_cloud]
                    else:
                        point_clouds[label] = point_cloud
    
    # Merge point clouds for each label
    merged_point_clouds = {}
    for label, clouds in point_clouds.items():
        if isinstance(clouds, list):
            # If multiple point clouds, merge them
            merged_cloud = clouds[0]
            for cloud in clouds[1:]:
                merged_cloud = merged_cloud.merge(cloud)
        else:
            # If single point cloud, use as is
            merged_cloud = clouds
        merged_point_clouds[label] = merged_cloud
    
    # visualize_point_clouds(merged_point_clouds)
    meshes = {}
    for label, point_cloud in merged_point_clouds.items():
        mesh = create_mesh_with_ball_pivoting(point_cloud, radii=[5, 10, 15, 20])
        
        # Visualize the mesh
        if label == 14:
            o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
        meshes[label] = mesh

    

    

    return merged_point_clouds
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process 3D instance volumes to midline volumes.')
    parser.add_argument('--test', action='store_true', help='Run in test mode (process only first 3 files)')
    parser.add_argument('--replace', action='store_true', help='Replace existing _mask_thinned.nrrd files')
    parser.add_argument('--time', action='store_true', help='Show execution time for each file')
    parser.add_argument('--filter-labels', action='store_true', help='Filter and reassign labels')
    parser.add_argument('--input-dir', type=str, help='Input directory path')
    parser.add_argument('--no-mask-out', action='store_true', help='Do not mask out values that arent part of the structure')
    args = parser.parse_args()


    default_input_directory = '/Users/jamesdarby/Desktop/manually_labelled_cubes/public_s1-8um'
    input_directory = args.input_dir if args.input_dir else default_input_directory
    label_values = None  # List of label values to process, pass None to process all labels
    overall_start_time = time.time()
    test_cubes= []
    if args.test:
        test_cubes = ['01744_02256_04048', '01744_02256_04304']

    process_directory(input_directory, replace=args.replace, test_cubes=test_cubes, show_time=args.time)
    
    if args.time:
        print(f"Total execution time: {time.time() - overall_start_time:.2f} seconds")