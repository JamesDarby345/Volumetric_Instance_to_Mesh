# Volumetric Instance to Midline Volume to .obj Mesh pipeline

## Purpose

This repository contains code to convert volumetrically labelled cubes into .obj files for flattening and rendering. The process involves taking an instance label, reducing it to a midline label (a 1 voxel thick representation), and then using that label to create an .obj file. The code also aims to maintain/create the manifold and non-intersecting mesh properties required for the autosegmentation grand prize from the volumetric instance labels.

## Install
graph-tool requires conda to install, so we need to setup a conda environment.
This assumes you have conda installed, if not, follow the instructions here: https://docs.anaconda.com/free/miniconda/miniconda-install/
```
conda env create -f environment.yml
conda activate vol_instance_to_mesh
```

## Steps

0. Change the paths in the scripts to point to the 3d volumetric instance nrrd file folder, the expected name is zyx_mask.nrrd as on the download server, within the zyx folder names. <br> Ex: 02000_02000_02000/02000_02000_02000_mask.nrrd
1. Run filter_and_reassign_labels.py to filter out small objects and reassign labels in the nrrd files, which will make running the next scripts faster.
2. Run a instance to midline volume conversion script, differences, pros and cons of each method are commented in the script and explained below.<br>
Update: Use fb_avg_3dinstance_to_midline_volume.py, it is the fastest and produces the best results now. By masking out voxels that left the original label followed by delauney2d, its downsides were negated.
3. Run the midline_to_obj.py file to convert the midline volume to .obj files.

If you want to visualise the data after any of these steps, I recommend dragging it into Napari, or into 3D Slicer (hit the crosshair icon to center the view on the data in slicer).

All outputs will be saved in the target folder <br>
Ex: <br>
02000_02000_02000/02000_02000_02000_fb_avg_mask_thinned.nrrd <br>
02000_02000_02000/obj/02000_02000_02000_fb_avg_mask_thinned_1.obj

## Instance to midline volume approaches

### Front-back average
File: fb_avg_3d_instance_to_midline_volume.py

**Update:** Masking voxels that left the original label and relying on delauney_2d triangulation to fill holes results in this method being the fastest and produces the best results. Consider the other two deprecated. This approach also reduces the chances of mesh collision even further.

Fastest midline calculation method.<br>
Front and back of the instance volume are averaged to create a midline volume.<br>
PCA components are used to determine the orientation of the structure.<br>
Doesn't natively fill holes.<br>
Disconnected fibres in the same label are an issue for this method as the
front-back average ends up out in space off of the sheet.<br>
Follows midline more closely on aggressive curves.<br>
Morphological tunnels are filled during the mesh creation process.<br>
Still leads to some rougher label connections than the graph approach, but is more representative of the shape<br>


**Typical Result:** <br>
<img width="700" alt="Screenshot 2024-09-03 at 10 30 04 AM" src="https://github.com/user-attachments/assets/c81ace58-3d02-4861-b839-bd439e62e209">
<img width="700" alt="Screenshot 2024-09-03 at 10 30 12 AM" src="https://github.com/user-attachments/assets/d2f9d636-5a57-4ee4-88d3-9724f1d0971f"><br>

**Outdated:** Typical unideal/'failure' case:<br>
<img width="700" alt="Screenshot 2024-08-31 at 2 39 40 PM" src="https://github.com/user-attachments/assets/2989888f-b31c-40e0-8a21-77da1b21e08d"><br>
Note how the mesh is averaging into empty space, outside the sheet

### Graph cut
File: graph_cut_3d_instance_to_midline_volume.py 

Slow due to graph construction which adds valid seam/sheet prior.<br>
Results in 'smoother' midlines, but less accurate to the original midline<br>
volume. When the label changes directions aggressively, the midline can<br>
end up curving away from the original midline volume.<br>
Single value assumption along maximum PCA direction.
Can only create voxels at maximum 45 degree angle to previous voxels,
limiting ability to follow aggressive curves.<br>
Fill holes (morphological tunnels) well by using the graph construction 
to provide structure across the gap.<br>
Could cause obj collisions as midline labels can leave their instance label
to allow for hole crossing, and end up inside of other objects.<br>

Biggest drawbacks are compute time and potential for resulting mesh collisions<br>
**Typical midline volume result:**<br>
<img width="777" alt="Screenshot 2024-08-30 at 5 50 44 PM" src="https://github.com/user-attachments/assets/fc816a89-d171-49e2-8a4c-613309879ed3"><br>
See how the volume is limited to 45 degree angles, which could result in the volume exiting the instance label in aggressive curves. Can be partially negated by running on smaller cubes/ROI's.

### Distance map
File: dist_map_3d_instance_to_midline_volume.py

Faster than graph construction method, slower than front-back average method.<br>
Doesn't fill holes (morphological tunnels).<br>
Constructs distance map for each label<br>
Uses single value assumption along maximum PCA direction to take highest dist map
value at each point to create the midline volume.<br>
Follows midline more closely on aggressive curves.<br>
Morphological tunnels are filled during the mesh creation process.<br>
Leads to closer midlines, but rougher labels, especially around holes.<br>
Should have a low chance of causing obj collisions as the midline cannot
leave the instance label.<br>

**Typical unideal/'failure' case:**<br>
<img width="571" alt="Screenshot 2024-08-31 at 11 29 50 AM" src="https://github.com/user-attachments/assets/907d4c8e-5ab1-42f6-b1a6-7ef13ef11c76"><br>
See how the resulting mesh is connected between different parts of the sheet, resulting in rougher labels.


## Discussion of Approach & Challenges
A key challenge is dealing with the holes, or more specifically the 'morphological tunnels' in the instance label.
This is mostly dealt with by assuming the sheet can be represented accurately with a single value on each x,y line (1 per z-axis). The code also computes the PCA components and rotates the individual labels such that they face as perpendicular to the z-axis as possible to reduce the impact of this assumption. The midline volume is then calculated by one of the 3 approaches. The graph approach deals with morphological tunnels by using the valid seam constraint introduced by the graph construction. The other approaches rely on filling the holes during the mesh construction step. This requires setting a maximum connection length parameter, and could result in U-shaped meshes etc being filled as if it is a hole. Though unideal, this should have minimal effect on the downstream flattening and rendering task, just resulting in some empty space being rendered.
