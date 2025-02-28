#!/usr/bin/env python3
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from shapely import wkt
from PIL import Image

def draw_image_with_labels(ax, image, json_data, use_xy=True):
    """
    Draws an image on a matplotlib axis with overlaid polygon labels.
    
    Parameters:
      ax        : A matplotlib axes object.
      image     : The image as a NumPy array.
      json_data : The loaded JSON data containing label information.
      use_xy    : If True, use pixel ("xy") coordinates; otherwise, use geographic ("lng_lat").
    """
    ax.imshow(image)
    coord_key = "xy" if use_xy else "lng_lat"
    
    for feature in json_data["features"][coord_key]:
        poly = wkt.loads(feature["wkt"])
        label = feature["properties"].get("subtype", "building")
        x, y = poly.exterior.xy
        ax.plot(x, y, color='red', linewidth=2)
        centroid = poly.centroid
        ax.text(centroid.x, centroid.y, label, color='yellow', fontsize=8,
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
    ax.axis('off')

def load_json(json_path):
    """
    Loads and returns JSON data from the given file path.
    """
    with open(json_path, 'r') as f:
        return json.load(f)

def visualize_side_by_side(disaster_folder, image_id, use_xy=True, show=True, save_path=None):
    """
    Visualizes pre- and post-disaster images side by side with their label overlays.
    
    Parameters:
      disaster_folder : Path to the folder containing 'images' and 'labels'.
      image_id        : Base image id (e.g., "guatemala-volcano_00000000").
      use_xy          : If True, use pixel coordinates; else, geographic coordinates.
      show            : If True, display the figure interactively.
      save_path       : If provided, save the figure to this file path.
    """
    images_dir = os.path.join(disaster_folder, "images")
    labels_dir = os.path.join(disaster_folder, "labels")
    
    # Build filenames for pre- and post-disaster images and JSONs.
    pre_img   = f"{image_id}_pre_disaster.png"
    post_img  = f"{image_id}_post_disaster.png"
    pre_json  = f"{image_id}_pre_disaster.json"
    post_json = f"{image_id}_post_disaster.json"

    pre_image_path  = os.path.join(images_dir, pre_img)
    post_image_path = os.path.join(images_dir, post_img)
    pre_json_path   = os.path.join(labels_dir, pre_json)
    post_json_path  = os.path.join(labels_dir, post_json)
    
    # Check that all files exist.
    for path in [pre_image_path, post_image_path, pre_json_path, post_json_path]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")

    # Load images and JSON label data.
    pre_image  = np.array(Image.open(pre_image_path))
    post_image = np.array(Image.open(post_image_path))
    pre_data   = load_json(pre_json_path)
    post_data  = load_json(post_json_path)
    
    # Create side-by-side figure.
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(f"Side-by-Side Visualization: {image_id}", fontsize=16)
    
    draw_image_with_labels(axs[0], pre_image, pre_data, use_xy)
    axs[0].set_title("Pre-Disaster")
    
    draw_image_with_labels(axs[1], post_image, post_data, use_xy)
    axs[1].set_title("Post-Disaster")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)

def get_image_ids(images_dir):
    """
    Scans the images directory and returns a sorted list of unique image IDs 
    (i.e. file names without the "_pre_disaster.png" suffix).
    """
    image_ids = set()
    for filename in os.listdir(images_dir):
        if filename.endswith("_pre_disaster.png"):
            base_id = filename.replace("_pre_disaster.png", "")
            image_ids.add(base_id)
    return sorted(image_ids)

def visualize_all_side_by_side(disaster_folder, use_xy=True, save_dir=None):
    """
    Iterates over all image IDs in a disaster folder and visualizes each side-by-side.
    
    Parameters:
      disaster_folder : Path to the folder containing 'images' and 'labels'.
      use_xy          : If True, use pixel coordinates; else, geographic coordinates.
      save_dir        : If provided, save each figure to this directory instead of displaying.
    """
    images_dir = os.path.join(disaster_folder, "images")
    image_ids = get_image_ids(images_dir)
    
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for image_id in image_ids:
        print(f"Visualizing: {image_id}")
        if save_dir:
            save_path = os.path.join(save_dir, f"{image_id}_side_by_side.png")
            visualize_side_by_side(disaster_folder, image_id, use_xy, show=False, save_path=save_path)
            print(f"Saved visualization to {save_path}")
        else:
            visualize_side_by_side(disaster_folder, image_id, use_xy)

if __name__ == "__main__":
    # Determine the project root (two levels up from scripts/visualization/)
    script_dir = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    
    # Set the disaster folder (adjust if you want to iterate over a different disaster)
    disaster_folder = os.path.join(project_root, "data", "xBD", "guatemala-volcano")
    
    # Uncomment one of the options below:
    
    # 1. Display all images interactively one by one.
    # visualize_all_side_by_side(disaster_folder, use_xy=True)
    
    # 2. Save all visualizations to a specified directory.
    save_directory = os.path.join(project_root, "experiments", "visualizations")
    visualize_all_side_by_side(disaster_folder, use_xy=True, save_dir=save_directory)