"""
Assumes that the labels have been synced with cube_label_reassignment already
Read in all the .obj files in a directory
merge ones that are connected into larger segments
output a new .obj file for each segment
MVP is boolean and the outputs with the same _## value and smooth them together?
"""

import os
from matplotlib import pyplot as plt
# from sklearn.neighbors import KDTree
import trimesh
import numpy as np
from tqdm import tqdm
import argparse
import pyvista as pv
from midline_helper_simplified import *
import time
import multiprocessing
from functools import partial
import open3d as o3d
from scipy.spatial import cKDTree

def find_obj_files(directory):
    obj_files = {}
    for root, dirs, files in os.walk(directory):
        if 'obj' in dirs:
            obj_folder = os.path.join(root, 'obj')
            z_y_x = os.path.basename(os.path.dirname(obj_folder))
            obj_files[z_y_x] = [os.path.join(obj_folder, file) for file in os.listdir(obj_folder) if file.endswith('.obj')]

    return obj_files

def merge_adjacent_meshes(meshes):
    # Merge the meshes
    result_mesh = trimesh.util.concatenate(meshes)
    
    # Remove duplicate vertices
    result_mesh.merge_vertices()
    
    # Remove duplicate and degenerate faces
    result_mesh.update_faces(result_mesh.unique_faces())
    result_mesh.update_faces(result_mesh.nondegenerate_faces())
    
    # Ensure consistent face winding
    result_mesh.fix_normals()
    
    # Fill holes (if any)
    trimesh.repair.fill_holes(result_mesh)
    
    return result_mesh

def parse_z_y_x(z_y_x_str):
    """Parse the z_y_x string into integer coordinates."""
    z, y, x = map(int, z_y_x_str.split('_'))
    return z, y, x

def are_adjacent(coord1, coord2, cube_size=256):
    """Determine if two coordinates are adjacent."""
    dz = abs(coord1[0] - coord2[0])
    dy = abs(coord1[1] - coord2[1])
    dx = abs(coord1[2] - coord2[2])
    return (dz + dy + dx) == cube_size  # Adjacent if exactly one coordinate differs by 1

def get_boundary_faces(mesh1, mesh2):
    """Return the list of the faces that make up the edge closest to the adjacent mesh"""
    # Find the closest points between the two meshes
    closest_points1, closest_points2 = trimesh.proximity.closest_point(mesh1, mesh2)
    
    # Find the faces that contain these closest points
    closest_faces1 = mesh1.faces[trimesh.proximity.points_face_indices(mesh1, closest_points1)]
    closest_faces2 = mesh2.faces[trimesh.proximity.points_face_indices(mesh2, closest_points2)]

    return closest_faces1, closest_faces2

def get_sheet_boundary(mesh):
    """
    Takes a mesh that is a sheet and returns the boundary faces as a new mesh.
    
    Args:
    mesh (trimesh.Trimesh): Input mesh representing a sheet.
    
    Returns:
    trimesh.Trimesh: A new mesh containing only the boundary faces.
    """
    # Replace the incorrect sum with bincount
    edge_counts = np.bincount(mesh.edges_unique_inverse)
    boundary_edges = mesh.edges_unique[edge_counts == 1]
    
    # Get the vertices of these boundary edges
    boundary_vertices = np.unique(boundary_edges)
    
    # Get the faces that contain these boundary vertices
    boundary_face_mask = np.isin(mesh.faces, boundary_vertices).any(axis=1)
    boundary_faces = mesh.faces[boundary_face_mask]
    
    # Create a new mesh with only the boundary faces
    boundary_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=boundary_faces)
    
    # Remove any isolated vertices
    boundary_mesh.remove_unreferenced_vertices()
    
    return boundary_mesh

def get_adjacent_boundary_faces(boundary_mesh1, boundary_mesh2, z_y_x_1, z_y_x_2, cube_size=256, threshold=2):
    """
    Find the adjacent faces of two boundary meshes given their cube coordinates.
    
    Args:
    boundary_mesh1, boundary_mesh2 (trimesh.Trimesh): Boundary meshes of two adjacent cubes
    z_y_x_1, z_y_x_2 (str): Cube coordinates in 'z_y_x' format
    cube_size (int): Size of each cube
    threshold (float): Distance threshold for considering faces as adjacent
    
    Returns:
    tuple: Two trimesh.Trimesh objects containing only the adjacent boundary faces
    """
    # Parse cube coordinates
    coord1 = parse_z_y_x(z_y_x_1)
    coord2 = parse_z_y_x(z_y_x_2)

    print(coord1, coord2)
    
    # Determine the adjacent face
    diff = np.array(coord2) - np.array(coord1)
    axis = np.argmax(np.abs(diff))
    is_positive = diff[axis] > 0

    # Check if faces are actually adjacent using cube_size
    if np.abs(diff[axis]) != cube_size or np.any(np.abs(diff[np.arange(3) != axis]) > 1e-6):
        print('no faces adjacent')
        print('np.abs(diff[axis]) != cube_size ', np.abs(diff[axis]) != cube_size, np.abs(diff[axis]))
        print('np.any(np.abs(diff[np.arange(3) != axis]) > 1e-6) ', np.any(np.abs(diff[np.arange(3) != axis]) > 1e-6))
        return []  # No faces are adjacent, return an empty list
    
    # Define the plane equation for the adjacent face
    plane_point = np.array(coord1)
    if is_positive:
        plane_point[axis] += cube_size
    plane_normal = np.zeros(3)
    plane_normal[axis] = 1 if is_positive else -1
    
    # Function to get faces close to the adjacent plane
    def get_close_faces(mesh, plane_point, plane_normal, threshold):
        # Calculate distances from all vertices to the plane
        distances = np.abs(np.dot(mesh.vertices - plane_point, plane_normal))
        # Find vertices close to the plane
        close_vertices = distances < threshold
        # Find faces that have at least one vertex close to the plane
        close_faces = np.any(close_vertices[mesh.faces], axis=1)
        return mesh.faces[close_faces]
    
    # Get close faces for both meshes
    close_faces1 = get_close_faces(boundary_mesh1, plane_point, plane_normal, threshold)
    close_faces2 = get_close_faces(boundary_mesh2, plane_point, -plane_normal, threshold)

    # Create new meshes with only the close faces
    adjacent_boundary1 = trimesh.Trimesh(vertices=boundary_mesh1.vertices, faces=close_faces1)
    adjacent_boundary2 = trimesh.Trimesh(vertices=boundary_mesh2.vertices, faces=close_faces2)
    
    # Remove unreferenced vertices
    adjacent_boundary1.remove_unreferenced_vertices()
    adjacent_boundary2.remove_unreferenced_vertices()
    
    return adjacent_boundary1, adjacent_boundary2

def clip_mesh_near_plane(mesh, plane_point, plane_normal, threshold):
    """
    Clips a mesh by removing faces and vertices within a threshold distance from a plane.

    Args:
        mesh (trimesh.Trimesh): The input mesh.
        plane_point (np.ndarray): A point on the plane.
        plane_normal (np.ndarray): The normal vector of the plane.
        threshold (float): The distance threshold for clipping.

    Returns:
        trimesh.Trimesh: The clipped mesh.
    """
    # Calculate distances from all vertices to the plane
    distances = np.dot(mesh.vertices - plane_point, plane_normal)

    # Find vertices close to the plane
    close_vertices = np.abs(distances) < threshold

    # Find faces that have at least one vertex close to the plane
    close_faces = np.any(close_vertices[mesh.faces], axis=1)

    # Remove close faces from the mesh
    mesh.faces = mesh.faces[~close_faces]

    # Remove unreferenced vertices
    mesh.remove_unreferenced_vertices()

    return mesh

def clip_adjacent_meshes(mesh1, mesh2, z_y_x_1, z_y_x_2, cube_size=256, threshold=5):
    """
    Clips both meshes by removing faces and vertices within a threshold distance from the adjacent plane.

    Args:
        mesh1 (trimesh.Trimesh): The first mesh.
        mesh2 (trimesh.Trimesh): The second mesh.
        z_y_x_1 (str): Cube coordinates for the first mesh in 'z_y_x' format.
        z_y_x_2 (str): Cube coordinates for the second mesh in 'z_y_x' format.
        cube_size (int, optional): Size of each cube. Defaults to 256.
        threshold (float, optional): Distance threshold for clipping. Defaults to 2.

    Returns:
        tuple: Two trimesh.Trimesh objects representing the clipped meshes.
    """
    # Create copies of the meshes to avoid modifying the originals
    mesh1_copy = mesh1.copy()
    mesh2_copy = mesh2.copy()

    coord1 = parse_z_y_x(z_y_x_1)
    coord2 = parse_z_y_x(z_y_x_2)

    if not are_adjacent(coord1, coord2, cube_size):
        print(f"Cubes {z_y_x_1} and {z_y_x_2} are not adjacent.")
        return mesh1_copy, mesh2_copy

    # Determine the adjacent axis and direction
    diff = np.array(coord2) - np.array(coord1)
    axis = np.argmax(np.abs(diff))
    is_positive = diff[axis] > 0

    # Define the plane for clipping
    plane_point = np.array(coord1)
    if is_positive:
        plane_point[axis] += cube_size
    else:
        plane_point[axis] -= cube_size
    plane_normal = np.zeros(3)
    plane_normal[axis] = 1 if is_positive else -1

    # Clip both copies of the meshes
    clipped_mesh1 = clip_mesh_near_plane(mesh1_copy, plane_point, plane_normal, threshold)
    clipped_mesh2 = clip_mesh_near_plane(mesh2_copy, plane_point, -plane_normal, threshold)

    return clipped_mesh1, clipped_mesh2

def clip_adjacent_meshes_pv(mesh1, mesh2, z_y_x_1, z_y_x_2, cube_size=256, threshold=5, center_threshold=10):
    """
    Clips two adjacent PyVista PolyData meshes near their adjacent plane, moving the threshold into each cube.

    Args:
        mesh1 (pyvista.PolyData): The first mesh.
        mesh2 (pyvista.PolyData): The second mesh.
        z_y_x_1 (str): Identifier for the first cube in 'z_y_x' format.
        z_y_x_2 (str): Identifier for the second cube in 'z_y_x' format.
        cube_size (int, optional): Size of each cube. Defaults to 256.
        threshold (float, optional): Distance threshold for clipping. Defaults to 5.

    Returns:
        tuple: Two clipped pyvista.PolyData meshes.
    """
    coord1 = parse_z_y_x(z_y_x_1)
    coord2 = parse_z_y_x(z_y_x_2)

    if not are_adjacent(coord1, coord2, cube_size):
        print(f"Cubes {z_y_x_1} and {z_y_x_2} are not adjacent.")
        return mesh1, mesh2

    diff = np.array(coord2) - np.array(coord1)
    axis = np.argmax(np.abs(diff))
    is_positive = diff[axis] > 0

    plane_origin = np.array(coord1, dtype=float)
    plane_normal = np.zeros(3)
    plane_normal[axis] = 1 if is_positive else -1

    if is_positive:
        plane_origin[axis] += cube_size
    else:
        plane_origin[axis] -= cube_size

    # Create two planes, one for each mesh, moved by the threshold
    plane1 = pv.Plane(center=plane_origin - plane_normal * threshold, direction=plane_normal, i_size=cube_size*2, j_size=cube_size*2)
    plane2 = pv.Plane(center=plane_origin + plane_normal * threshold, direction=plane_normal, i_size=cube_size*2, j_size=cube_size*2)
    plane3 = pv.Plane(center=plane_origin - plane_normal * center_threshold, direction=plane_normal, i_size=cube_size*2, j_size=cube_size*2)
    plane4 = pv.Plane(center=plane_origin + plane_normal * center_threshold, direction=plane_normal, i_size=cube_size*2, j_size=cube_size*2)
    plane5 = pv.Plane(center=plane_origin - plane_normal * (center_threshold - 1), direction=plane_normal, i_size=cube_size*2, j_size=cube_size*2)
    plane6 = pv.Plane(center=plane_origin + plane_normal * (center_threshold - 1), direction=plane_normal, i_size=cube_size*2, j_size=cube_size*2)

    clipped_mesh1 = mesh1.clip_surface(plane1, invert=True)
    clipped_mesh2 = mesh2.clip_surface(plane2, invert=False)
    center_clipped_mesh = clipped_mesh1.clip_surface(plane3, invert=False) + clipped_mesh2.clip_surface(plane4, invert=True)
    clipped_mesh1 = mesh1.clip_surface(plane5, invert=True)
    clipped_mesh2 = mesh2.clip_surface(plane6, invert=False)

    return clipped_mesh1, clipped_mesh2, center_clipped_mesh

def clip_mesh_near_adjacent_plane_pv(z_y_x_1, z_y_x_2, mesh, cube_size=256, threshold=5):
    """
    Clips the input mesh to retain only the portion within a specified threshold
    distance from the adjacent plane between two cubes defined by their ZYX coordinates.

    Args:
        z_y_x_1 (str): Identifier for the first cube in 'z_y_x' format.
        z_y_x_2 (str): Identifier for the second cube in 'z_y_x' format.
        mesh (pyvista.PolyData): The input mesh to be clipped.
        cube_size (int, optional): Size of each cube. Defaults to 256.
        threshold (float, optional): Distance threshold for clipping. Defaults to 5.

    Returns:
        pyvista.PolyData: The clipped mesh containing only the section within the threshold.
    """
    coord1 = parse_z_y_x(z_y_x_1)
    coord2 = parse_z_y_x(z_y_x_2)

    if not are_adjacent(coord1, coord2, cube_size):
        print(f"Cubes {z_y_x_1} and {z_y_x_2} are not adjacent.")
        return None

    diff = np.array(coord2) - np.array(coord1)
    axis = np.argmax(np.abs(diff))
    is_positive = diff[axis] > 0

    plane_origin = np.array(coord1, dtype=float)
    plane_normal = np.zeros(3)
    plane_normal[axis] = 1 if is_positive else -1

    if is_positive:
        plane_origin[axis] += cube_size
    else:
        plane_origin[axis] -= cube_size

    # Create a plane for clipping
    plane1 = pv.Plane(center=plane_origin - plane_normal * threshold, direction=plane_normal, i_size=cube_size*2, j_size=cube_size*2)
    plane2 = pv.Plane(center=plane_origin + plane_normal * threshold, direction=plane_normal, i_size=cube_size*2, j_size=cube_size*2)

    # Clip the mesh with the plane and threshold
    clipped_mesh = mesh.clip_surface(plane1, invert=False) 
    clipped_mesh = clipped_mesh.clip_surface(plane2, invert=True)

    return clipped_mesh

def extract_section_near_adjacent_plane(z_y_x_1, z_y_x_2, mesh, cube_size=256, threshold=5):
    """
    Extracts a section of the mesh within a threshold distance from the adjacent plane between two cubes.

    Args:
        z_y_x_1 (str): First cube coordinates in 'z_y_x' format.
        z_y_x_2 (str): Second cube coordinates in 'z_y_x' format.
        mesh (trimesh.Trimesh): The input mesh to extract the section from.
        cube_size (int, optional): Size of each cube. Defaults to 256.
        threshold (float, optional): Distance threshold for extraction. Defaults to 5.

    Returns:
        trimesh.Trimesh: The extracted and cleaned mesh section.
    """
    coord1 = parse_z_y_x(z_y_x_1)
    coord2 = parse_z_y_x(z_y_x_2)

    if not are_adjacent(coord1, coord2, cube_size):
        print(f"Cubes {z_y_x_1} and {z_y_x_2} are not adjacent.")
        return None

    diff = np.array(coord2) - np.array(coord1)
    axis = np.argmax(np.abs(diff))
    is_positive = diff[axis] > 0

    # Define the plane for extraction
    plane_point = np.array(coord1, dtype=float)
    if is_positive:
        plane_point[axis] += cube_size
    plane_normal = np.zeros(3)
    plane_normal[axis] = 1 if is_positive else -1

    # Calculate signed distances from all vertices to the plane
    distances = np.dot(mesh.vertices - plane_point, plane_normal)

    # Identify vertices within the threshold on both sides of the plane
    mask = np.abs(distances) <= threshold
    if not np.any(mask):
        print("No vertices found within the threshold distance from the plane.")
        return None

    # Select faces where all vertices are within the threshold
    selected_faces = mesh.faces[np.all(mask[mesh.faces], axis=1)]

    if selected_faces.size == 0:
        print("No faces found within the threshold distance from the plane.")
        return None

    # Create the new mesh with the selected faces
    cleaned_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=selected_faces)
    cleaned_mesh.remove_unreferenced_vertices()

    return cleaned_mesh

def merge_adjacent_cubes(z_y_x_1, z_y_x_2, obj_files, output_directory, cube_size=256):
    coord1 = parse_z_y_x(z_y_x_1)
    coord2 = parse_z_y_x(z_y_x_2)
    
    if not are_adjacent(coord1, coord2, cube_size):
        print(f"Cubes {z_y_x_1} and {z_y_x_2} are not adjacent.")
        return {}
    
    cube1_paths = obj_files[z_y_x_1]
    cube2_paths = obj_files[z_y_x_2]
        
    for file1 in cube1_paths:
        label1 = file1.split('_')[-1].split('.')[0]
        for file2 in cube2_paths:
            label2 = file2.split('_')[-1].split('.')[0]
            
            # if label values are the same, then this is the same sheet; merge them
            if label1 == label2:
                clip_threshold = 5
                center_threshold = 20

                pv_mesh1 = pv.read(file1)
                pv_mesh2 = pv.read(file2)

                pv_clipped_mesh_1, pv_clipped_mesh_2, pv_center_clipped_mesh = clip_adjacent_meshes_pv(pv_mesh1, pv_mesh2, z_y_x_1, z_y_x_2, cube_size=256, threshold=clip_threshold, center_threshold=center_threshold)

                # center_temp = pyvista_to_trimesh(pv_center_clipped_mesh, avg_pca_normal=None, should_print_timing=False, should_fix_normals=False)
                # center_temp.export(os.path.join(output_directory, 'test_center.obj'))
                # test_center_largest = pv_center_clipped_mesh.extract_largest(inplace=False)
                
                # temp = pyvista_to_trimesh(test_center_largest, avg_pca_normal=None, should_print_timing=False, should_fix_normals=False)
                # temp.export(os.path.join(output_directory, 'test_center_largest.obj'))
                # Convert the combined mesh to a surface using Delaunay triangulation
                surface = pv_center_clipped_mesh.delaunay_2d(alpha=2*clip_threshold)
                surface = filter_disconnected_parts(surface, min_vertices=800)
                if surface.n_points == 0:
                    # print(f"No points left after filtering and clipping, skipping mesh creation for {z_y_x_1} and {z_y_x_2} label {label1}")
                    continue

                # Fill holes in the surface mesh, align normals
                surface = surface.delaunay_2d(alpha=8*clip_threshold)
                surface = surface.fill_holes(hole_size=12*clip_threshold)
                surface.compute_normals(auto_orient_normals=True, inplace=True)

                surface = clip_mesh_near_adjacent_plane_pv(z_y_x_1, z_y_x_2, surface, cube_size=256, threshold=center_threshold-1)
                combined_mesh = pv.merge([pv_clipped_mesh_1, pv_clipped_mesh_2, surface])

                # Remove duplicated vertices and clean up the mesh
                cleaned_mesh = combined_mesh.clean(
                    tolerance=1e-6,  # Adjust tolerance as needed, too high and it is no longer triangulated and ends up with holes
                    point_merging=True,
                    inplace=True
                )

                if not cleaned_mesh.is_all_triangles:
                    # print("Mesh is not triangulated, triangulating...")
                    cleaned_mesh = cleaned_mesh.triangulate()
                    # Fill holes in the mesh; not reliable due to morphological tunnel vs hole
                    cleaned_mesh = cleaned_mesh.fill_holes(hole_size=20)  # Adjust hole_size as needed
                cleaned_mesh.compute_normals(auto_orient_normals=True, inplace=True) #much faster than trimesh to align normals
                # Extract the largest connected component
                lmesh = cleaned_mesh.extract_largest(inplace=False)
                # cleaned_mesh = largest_component
                # Save the cleaned mesh as an .obj file
                tm_mesh = pyvista_to_trimesh(lmesh, avg_pca_normal=None, should_print_timing=False, should_fix_normals=False)
                output_path = os.path.join(output_directory, f'sheet_val_{label1}')
                os.makedirs(output_path, exist_ok=True)
                tm_mesh.export(os.path.join(output_path, f'{z_y_x_1}-{z_y_x_2}_mesh.obj'))
                # print(f"Saved mesh {z_y_x_1}-{z_y_x_2}_mesh.obj")
    

def process_cube_pair(zyx_pair, obj_files, output_directory, cube_size=256):
    zyx_1, zyx_2 = zyx_pair
    start_time = time.time()
    merge_adjacent_cubes(zyx_1, zyx_2, obj_files, output_directory, cube_size)
    print(f"Time taken to merge {zyx_1} and {zyx_2}: {time.time() - start_time} seconds")

def merge_adjacent_cubes_list_parallel(zyx_list, obj_files, output_directory, cube_size=256):
    """
    Merges adjacent cubes for a list of ZYX coordinates in parallel.

    Args:
    zyx_list (list): List of ZYX coordinates as strings (e.g., ['01744_02000_04048', '01744_02000_04304', ...])
    obj_files (dict): Dictionary of obj files keyed by ZYX coordinates
    output_directory (str): Path to the output directory
    cube_size (int): Size of each cube (default: 256)
    """
    adjacent_pairs = []
    for i in range(len(zyx_list)):
        for j in range(i + 1, len(zyx_list)):
            zyx_1 = zyx_list[i]
            zyx_2 = zyx_list[j]
            
            coord1 = parse_z_y_x(zyx_1)
            coord2 = parse_z_y_x(zyx_2)
            
            if are_adjacent(coord1, coord2, cube_size):
                adjacent_pairs.append((zyx_1, zyx_2))

    # Create a partial function with fixed arguments
    process_pair = partial(process_cube_pair, obj_files=obj_files, output_directory=output_directory, cube_size=cube_size)

    # Use multiprocessing to parallelize the processing
    num_cores = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_cores) as pool:
        list(tqdm(pool.imap(process_pair, adjacent_pairs), total=len(adjacent_pairs), desc="Merging adjacent cubes"))

def naive_merge_and_clean(output_directory, values=None):
    """
    For every folder named 'sheet_val_#' in the output directory, load all the .obj files,
    merge them using PyVista, clean the merged mesh, fill additional boundaries, and save it back as 'sheet_#_final_mesh.obj'.
    
    Args:
        output_directory (str): Path to the directory containing 'sheet_val_#' folders.
        values (list, optional): List of sheet numbers to process. If None, process all folders.
    """
    import os
    import re
    import pyvista as pv
    import open3d as o3d
    import numpy as np

    # Regular expression to match folders named 'sheet_val_#'
    sheet_val_pattern = re.compile(r'^sheet_val_(\d+)$')
    
    # Iterate through each item in the output directory
    for item in os.listdir(output_directory):
        match = sheet_val_pattern.match(item)
        if match:
            sheet_number = match.group(1)
            
            # If values is provided, skip folders not in the list
            if values and sheet_number not in values:
                continue
            
            sheet_folder = os.path.join(output_directory, item)
            
            if not os.path.isdir(sheet_folder):
                print(f"Skipping {sheet_folder}, not a directory.")
                continue
            
            # List all .obj files in the sheet folder
            obj_files = [f for f in os.listdir(sheet_folder) if f.endswith('_mesh.obj') and not f.endswith('_final_mesh.obj')]
            
            if not obj_files:
                print(f"No .obj files found in {sheet_folder}.")
                continue
            
            # Load all .obj files using PyVista
            meshes = []
            for obj_file in obj_files:
                obj_path = os.path.join(sheet_folder, obj_file)
                try:
                    mesh = pv.read(obj_path)
                    meshes.append(mesh)
                except Exception as e:
                    print(f"Failed to load {obj_path}: {e}")
            
            if not meshes:
                print(f"No valid meshes to merge in {sheet_folder}.")
                continue
            
            # Merge all meshes
            merged_mesh = pv.merge(meshes)
            merged_mesh = merged_mesh.subdivide_adaptive(max_edge_len=2, inplace=True)

            print(merged_mesh.n_points)
            # Clean the merged mesh
            merged_mesh = merged_mesh.clean(
                tolerance=5,
                point_merging=True,
                inplace=True
            )

            print(merged_mesh.n_points)
           
            point_cloud = merged_mesh.points

            pv_point_cloud_polydata = pv.PolyData(point_cloud)
            pv_point_cloud_polydata.plot(point_size=10)

            
            # Create a KDTree for efficient nearest neighbor search
            kdtree = cKDTree(point_cloud)

            # Define a radius for the neighborhood
            radius = 50.0  # Adjust this value based on your data

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
                indices_within_radius = kdtree.query_ball_point(point_cloud[current_index], r=radius/1.05)

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
                # Visualize the merged mesh
                merged_mesh.plot(color='lightblue', show_edges=True)

                merged_mesh_tm = pyvista_to_trimesh(merged_mesh, avg_pca_normal=None, should_print_timing=False, should_fix_normals=False)
                final_mesh_path = os.path.join(sheet_folder, f'sheet_{sheet_number}_final_mesh.obj')
                merged_mesh_tm.export(final_mesh_path)
            else:
                print("No local meshes were created.")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge meshes across nrrd cubes.')
    parser.add_argument("--input-path", type=str, help="Path to the directory containing mask files")
    parser.add_argument("--output-path", type=str, help="Path to the directory to save relabeled mask files into")
    parser.add_argument("--nm", action='store_true', help='Run naive merge and clean on the output directory')
    current_directory = os.getcwd()
    
    args = parser.parse_args()
    default_input_directory = '/Users/jamesdarby/Desktop/manually_labelled_cubes/public_s1-8um'
    input_directory = args.input_path if args.input_path else default_input_directory
    output_directory = args.output_path if args.output_path else os.path.join(current_directory, 'merged_meshes')

    os.makedirs(output_directory, exist_ok=True)

    obj_files = find_obj_files(input_directory)
    files_to_merge = ['01744_02000_04048', '01744_02000_04304', '01744_02256_04048', '01744_02256_04304']
    # files_to_merge = list(obj_files.keys())
    if not args.nm:
        merge_adjacent_cubes_list_parallel(files_to_merge, obj_files, output_directory, cube_size=256)
    if args.nm:
        naive_merge_and_clean(output_directory, ['69'])