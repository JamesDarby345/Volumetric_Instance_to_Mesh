# Instance to Midline Volume Converter

## Purpose

This repository contains code to convert volumetrically labelled cubes into .obj files for flattening and rendering. The process involves taking an instance label, reducing it to a midline label (a 1 voxel thick representation), and then using that label to create an .obj file. The code also aims to maintain/create the manifold and non-intersecting properties required for the autosegmentation grand prize from the instance labels.

## Install
Setup a virtual environment and pip install requirements.txt
(Note that conda is required to easily install graph-tool, if you want to use the graph cut approach)
```
pip install -r requirements.txt 
```

for graph-tool:
```
conda install -c conda-forge graph-tool
```

## Steps

0. Change the paths in the scripts to point to the 3d volumetric instance nrrd file folder, the expected name is zyx_mask.nrrd as on the download server, within the zyx folder names. <br> Ex: 02000_02000_02000/02000_02000_02000_mask.nrrd
1. Run filter_and_reassign_labels.py to filter out small objects and reassign labels in the nrrd files,which will make running the next scripts faster.
2. Run a instance to midline volume conversion script, differences, pros and cons of each method are commented in the script and explained below.
3. Run the midline_to_obj.py file to convert the midline volume to .obj files.

If you want to visualise the data after any of these steps, I recommend dragging it into 3D Slicer, and hit the crosshair icon to center the view on the data.

All outputs will be saved in the target folder <br>
Ex: <br>
02000_02000_02000/02000_02000_02000_fb_avg_mask_thinned.nrrd <br>
02000_02000_02000/obj/02000_02000_02000_fb_avg_mask_thinned_1.obj

## Instance to midline volume approaches

### Graph cut
File: graph_cut_3d_instance_to_midline_volume.py 

Slow due to graph construction which adds valid seam/sheet prior.<br>
Results in 'smoother' midlines, but less accurate to the original midline<br>
volume. When the label changes directions aggresively, the midline can<br>
end up curving away from the original midline volume.<br>
Single value assumption along maximum PCA direction.
Can only create voxels at maximum 45 degree angle to previous voxels,
limiting ability to follow aggresive curves.<br>
Fill holes (morphological tunnels) well by using the graph construction 
to provide structure across the gap.<br>
Could cause obj collisions as midline labels can leave their instance label
to allow for hole crossing, and end up inside of other objects.<br>

### Distance map
File: dist_map_3d_instance_to_midline_volume.py

Faster than graph construction method, slower than front back average method.<br>
Doesnt fill holes (morphological tunnels).<br>
Constructs distance map for each label<br>
Uses single value assumption along maximum PCA direction to take highest dist map
value at each point to create the midline volume.<br>
Follows midline more closely on aggresive curves.<br>
Morphological tunnels are filled during the mesh creation process.<br>
Leads to closer midlines, but rougher labels, especially around holes.<br>
Should have a low chance of causing obj collisions as the midline cannot
leave the instance label.<br>

### Front back average
File: fb_avg_3d_instance_to_midline_volume.py

Fastest midline calculation method.<br>
Front and back of the instance volume are averaged to create a midline volume.<br>
PCA components are used to determine the orientation of the structure.<br>
Doesnt natively fill holes.<br>
Disconnected fibres in the same label are an issue for this method as the
front back average ends up out in space off of the sheet.<br>
Follows midline more closely on aggresive curves.<br>
Morphological tunnels are filled during the mesh creation process.<br>
Leads to closer midlines, but rougher labels, especially around holes.<br>
The front back average could leave the instance label, causing obj collisions 
with low probability.<br>
