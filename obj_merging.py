"""
Assumes that the labels have been synced with cube_label_reassignment already
Read in all the .obj files in a directory
merge ones that are connected into larger segments
output a new .obj file for each segment
MVP is boolean and the outputs with the same _## value and smooth them together?
"""

import os
import trimesh
import numpy as np
from tqdm import tqdm
import argparse
import scipy.spatial
import pyvista as pv
from midline_helper_simplified import *

def find_obj_files(directory):
    obj_files = {}
    for root, dirs, files in os.walk(directory):
        if 'obj' in dirs:
            obj_folder = os.path.join(root, 'obj')
            z_y_x = os.path.basename(os.path.dirname(obj_folder))
            obj_files[z_y_x] = [os.path.join(obj_folder, file) for file in os.listdir(obj_folder) if file.endswith('.obj')]

    return obj_files

def merge_adjacent_meshes(mesh1, mesh2):
    # Merge the meshes
    result_mesh = trimesh.util.concatenate([mesh1, mesh2])
    
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

def clip_adjacent_meshes_pv(mesh1, mesh2, z_y_x_1, z_y_x_2, cube_size=256, threshold=5):
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

    clipped_mesh1 = mesh1.clip_surface(plane1, invert=True)
    clipped_mesh2 = mesh2.clip_surface(plane2, invert=False)

    return clipped_mesh1, clipped_mesh2

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
    
    boundary_faces = {}
    
    for file1 in cube1_paths:
        label1 = file1.split('_')[-1].split('.')[0]
        for file2 in cube2_paths:
            label2 = file2.split('_')[-1].split('.')[0]
            
            if label1 == label2:
                clip_threshold = 5
                mesh1 = trimesh.load_mesh(file1)
                mesh2 = trimesh.load_mesh(file2) 
                mesh1.export(os.path.join(output_directory, f'{label1}_mesh1.obj'))
                mesh2.export(os.path.join(output_directory, f'{label1}_mesh2.obj'))

                pv_mesh1 = pv.read(file1)
                pv_mesh2 = pv.read(file2)

                pv_clipped_mesh_1, pv_clipped_mesh_2 = clip_adjacent_meshes_pv(pv_mesh1, pv_mesh2, z_y_x_1, z_y_x_2, cube_size=256, threshold=clip_threshold)
                clipped_mesh_1 = pyvista_to_trimesh(pv_clipped_mesh_1, avg_pca_normal=None, should_print_timing=False, should_fix_normals=False)
                clipped_mesh_2 = pyvista_to_trimesh(pv_clipped_mesh_2, avg_pca_normal=None, should_print_timing=False, should_fix_normals=False)
                clipped_mesh_1.export(os.path.join(output_directory, f'{label1}_clipped_mesh1.obj'))
                clipped_mesh_2.export(os.path.join(output_directory, f'{label1}_clipped_mesh2.obj'))
                
                # combined_mesh = pv.merge([pv_mesh1, pv_mesh2])
                combined_mesh = pv.merge([pv_clipped_mesh_1, pv_clipped_mesh_2])

                print(f"Data type of combined_mesh: {type(combined_mesh)}")

                # Convert the combined mesh to a surface using Delaunay triangulation
                surface = combined_mesh.delaunay_2d(alpha=2*clip_threshold)
                surface = filter_disconnected_parts(surface, min_vertices=800)
                if surface.n_points == 0:
                    print("No points left after filtering, skipping mesh creation.")
                    continue
                surface = surface.delaunay_2d(alpha=8*clip_threshold)
                # Fill holes in the surface mesh
                surface = surface.fill_holes(hole_size=12*clip_threshold)
                surface.compute_normals(auto_orient_normals=True, inplace=True)

                #will need to calculate the average pca normal if we want to fix the normals
                tm_mesh = pyvista_to_trimesh(surface, avg_pca_normal=None, should_print_timing=False, should_fix_normals=False)
                tm_mesh.export(os.path.join(output_directory, f'{label1}_tm_mesh.obj'))

                clipped_mesh_1, clipped_mesh_2 = clip_adjacent_meshes(mesh1, mesh2, z_y_x_1, z_y_x_2, cube_size=256, threshold=clip_threshold)
                # clipped_mesh_1.export(os.path.join(output_directory, f'{label1}_clipped_mesh1.obj'))
                # clipped_mesh_2.export(os.path.join(output_directory, f'{label1}_clipped_mesh2.obj'))

                merged_mesh = merge_adjacent_meshes(clipped_mesh_1, clipped_mesh_2) #boolean and merge...
                merged_mesh.export(os.path.join(output_directory, f'{label1}_merged.obj'))

                delaunay = scipy.spatial.Delaunay(merged_mesh.vertices[:, :2])
                delaunay_faces = delaunay.simplices
                delaunay_mesh = trimesh.Trimesh(vertices=merged_mesh.vertices, faces=delaunay_faces)
                delaunay_mesh.export(os.path.join(output_directory, f'{label1}_delaunay.obj')) #very spikey results and messes up other parts of the mesh, but good near the boundary

                section_mesh = extract_section_near_adjacent_plane(z_y_x_1, z_y_x_2, delaunay_mesh, threshold=clip_threshold+5) #TODO if this threshold value is too small, it misses some of the mesh...
                section_mesh.export(os.path.join(output_directory, f'{label1}_middle_section.obj'))

                middle_section_merged_mesh = merge_adjacent_meshes(section_mesh, merged_mesh)
                middle_section_merged_mesh.export(os.path.join(output_directory, f'{label1}_middle_section_merged.obj'))

                # Calculate boundary meshes from the clipped meshes
                boundary_mesh_1 = get_sheet_boundary(clipped_mesh_1)
                boundary_mesh_2 = get_sheet_boundary(clipped_mesh_2)
                boundary_mesh_1, boundary_mesh_2 = get_adjacent_boundary_faces(boundary_mesh_1, boundary_mesh_2, z_y_x_1, z_y_x_2, cube_size=256, threshold=clip_threshold+2)
                # Save the boundary meshes
                boundary_mesh_1_path = os.path.join(output_directory, f'{label1}_boundary_1.obj')
                boundary_mesh_2_path = os.path.join(output_directory, f'{label1}_boundary_2.obj')
                boundary_mesh_1.export(boundary_mesh_1_path)
                boundary_mesh_2.export(boundary_mesh_2_path)
                print(f"Boundary meshes saved to: {boundary_mesh_1_path} and {boundary_mesh_2_path}")

                

                # Merge the two boundary meshes using Delaunay triangulation
                combined_vertices = np.vstack((boundary_mesh_1.vertices, boundary_mesh_2.vertices))
                combined_faces = np.vstack((boundary_mesh_1.faces, boundary_mesh_2.faces + len(boundary_mesh_1.vertices)))

                # Perform Delaunay triangulation on the combined vertices
                tri = trimesh.Trimesh(vertices=combined_vertices, faces=combined_faces)
                delaunay = scipy.spatial.Delaunay(tri.vertices[:, :2])
                delaunay_faces = delaunay.simplices

                # Create a new mesh with the Delaunay faces
                merged_boundary_mesh = trimesh.Trimesh(vertices=tri.vertices, faces=delaunay_faces)

                # Save the merged boundary mesh
                merged_boundary_mesh_path = os.path.join(output_directory, f'{label1}_merged_boundary.obj')
                merged_boundary_mesh.export(merged_boundary_mesh_path) #creates faces that are too large and doesnt merge the meshes together well
                print(f"Merged boundary mesh saved to: {merged_boundary_mesh_path}")
    
    return boundary_faces

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge meshes across nrrd cubes.')
    parser.add_argument("--input-path", type=str, help="Path to the directory containing mask files")
    parser.add_argument("--output-path", type=str, help="Path to the directory to save relabeled mask files into")

    current_directory = os.getcwd()
    
    args = parser.parse_args()
    default_input_directory = '/Users/jamesdarby/Desktop/manually_labelled_cubes/public_s1-8um'
    input_directory = args.input_path if args.input_path else default_input_directory
    output_directory = args.output_path if args.output_path else os.path.join(current_directory, 'merged_meshes')

    os.makedirs(output_directory, exist_ok=True)

    obj_files = find_obj_files(input_directory)
    files_to_merge = ['01744_02000_04048', '01744_02000_04304']
    # for file in files_to_merge:
    #     print(file)
    #     print(obj_files[file])
    #     print()

    merge_adjacent_cubes(files_to_merge[0], files_to_merge[1], obj_files, output_directory, cube_size=256)
