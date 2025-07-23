# This file visualizes the Poisson encoding of an arbitrary greyscale picture, using the poisson_fire function.   
# To try it out on a fixed PNG file, run "python visualizations/encoding_viz.py" at the root of the project. 

import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import sys
from PIL import Image
import os
sys.path.append("/Users/AidenZhou/Desktop/CS/Projects/SpikingNN/") # obviously, change this to your own path
from utils.data_encoding import poisson_fire

# This function loads a picture, converts it to greyscale, and resizes it to the target size. 
# It then returns a flattened list of pixel values. 

def load_image(image_path, target_size=(14, 14)):
    try:
        img = Image.open(image_path)
        
        # Convert to greyscale if not already
        if img.mode != 'L':
            img = img.convert('L')
        
        # Resize to target size
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to array and flatten
        img_array = np.array(img)
        flattened = img_array.flatten().tolist()
        
        return flattened, img_array
        
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None
    
# This function visualizes the Poisson encoding of an image. 
# It runs multiple trials of Poisson encoding, and visualizes the firing rates of each pixel. 

def visualize_encoding(image_path, num_trials=1000, max_rate=100, target_size=(500,500)):

    # Load and preprocess image
    im, img_array = load_image(image_path, target_size)
    
    num_firings = [0] * len(im)

    # Run multiple trials of Poisson encoding
    for i in range(num_trials):
        firings = poisson_fire(im, max_rate=max_rate)
        for j in range(len(im)):
            if firings[j]:
                num_firings[j] += 1

    # Create triple visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Original image
    ax1.imshow(img_array, cmap='gray_r')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Greyscale reconstruction (normalized firing rates)
    # Normalize firing rates to [0, 255] range for greyscale display
    max_firing = max(num_firings) if max(num_firings) > 0 else 1
    normalized_firing = np.array(num_firings) * 255 / max_firing
    reconstruction_array = np.reshape(normalized_firing, target_size).astype(np.uint8)
    
    ax2.imshow(reconstruction_array, cmap='gray_r')
    ax2.set_title('Greyscale Reconstruction')
    ax2.axis('off')
    
    # Firing rate visualization
    firing_array = np.reshape(num_firings, target_size)
    im3 = ax3.imshow(firing_array, cmap='hot')
    ax3.set_title(f'Firing Rates (avg: {np.mean(num_firings):.1f} spikes)')
    ax3.axis('off')
    
    # Add colorbar for firing rates
    plt.colorbar(im3, ax=ax3, label='Number of spikes')
    
    plt.tight_layout()
    plt.show()

# Usage
if __name__ == "__main__":
    image_path = "./data/brain_icon.png" # Change this to the path of your image
    
    if image_path and os.path.exists(image_path):
        visualize_encoding(image_path, target_size=(100, 100))
    else:
        print("No image path specified or file not found.")