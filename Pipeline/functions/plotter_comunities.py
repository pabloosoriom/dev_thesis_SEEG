import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import nibabel as nib
from nilearn import datasets
from PIL import Image

def plot_electrode_locations(xyz_loc, communities_data, outputpath, band):
    # Prepare the output directory
    images_dir = os.path.join(outputpath, "brain images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Load fsaverage template (high-res pial surface)
    fsaverage = datasets.fetch_surf_fsaverage()
    left_mesh_file = fsaverage.pial_left  # Left hemisphere
    right_mesh_file = fsaverage.pial_right  # Right hemisphere

    # Load the meshes using nibabel
    left_mesh = nib.load(left_mesh_file)
    right_mesh = nib.load(right_mesh_file)

    left_coords = left_mesh.darrays[0].data
    left_faces = left_mesh.darrays[1].data
    right_coords = right_mesh.darrays[0].data
    right_faces = right_mesh.darrays[1].data

    # Process each time step in communities_data
    for t, (algorithm, community_list, density) in enumerate(tqdm(communities_data, desc="Generating frames")):
        # Create 3D plot
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')

        # Plot left hemisphere
        ax.plot_trisurf(left_coords[:, 0], left_coords[:, 1], left_coords[:, 2],
                        triangles=left_faces, color='lightgray', alpha=0.1)

        # Plot right hemisphere
        ax.plot_trisurf(right_coords[:, 0], right_coords[:, 1], right_coords[:, 2],
                        triangles=right_faces, color='lightgray', alpha=0.5)

        # Plot electrodes
        if density == 0 or not community_list[0]:  # Plot all electrodes in blue if no community info
            title_text = f'Time {t}: - No Info'
            color = 'blue'
            for i, row in xyz_loc.iterrows():
                ax.scatter(row['r'], row['a'], row['s'], color=color, s=50)
                ax.text(row['r'], row['a'], row['s'], row['formatted_label'], color='black')
        else:
            title_text = f'Time {t}: {algorithm}'
            community = community_list[0]  # Extract the community list for this time step
            for i, row in xyz_loc.iterrows():
                color = 'red' if row['formatted_label'] in community else 'blue'
                ax.scatter(row['r'], row['a'], row['s'], color=color, s=50)
                ax.text(row['r'], row['a'], row['s'], row['formatted_label'], color='black')

        # Set axis labels and title
        ax.set_xlabel('Right (mm)')
        ax.set_ylabel('Anterior (mm)')
        ax.set_zlabel('Superior (mm)')
        ax.set_title(title_text)

        # Adjust view and plot settings
        ax.view_init(azim=150, elev=70)
        ax.grid(False)

        # Save the frame
        frame_path = os.path.join(images_dir, f"frame_{t:04d}.png")
        plt.savefig(frame_path)
        plt.close(fig)  # Close the figure to free memory

    # Create an animation from saved frames
    frames = [Image.open(os.path.join(images_dir, f"frame_{t:04d}.png")) for t in range(len(communities_data))]
    animation_path = os.path.join(outputpath, f"electrode_locations_{band}.gif")
    frames[0].save(animation_path, save_all=True, append_images=frames[1:], duration=500, loop=0)
    print("Animation saved at:", animation_path)
