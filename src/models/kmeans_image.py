from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans

# 1. Define paths relative to the project root
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / 'src' / 'data'
OUT = ROOT / 'reports' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

# 2. Check for the input image (PNG or JPG)
png_path = DATA_DIR / 'input_image.png'
jpg_path = DATA_DIR / 'input_image.jpg'

if png_path.exists():
    IMAGE_PATH = png_path
elif jpg_path.exists():
    IMAGE_PATH = jpg_path
else:
    raise FileNotFoundError("Could not find input_image.png or input_image.jpg in src/data/")

def run():
    print(f"Loading image from: {IMAGE_PATH}")
    
    # 3. Load image and convert to RGB (handles PNG transparency)
    img = Image.open(IMAGE_PATH).convert('RGB').resize((300, 300))
    arr = np.array(img)
    
    # Reshape 3D image matrix (300, 300, 3) to 2D matrix of pixels (90000, 3)
    X = arr.reshape(-1, 3).astype(float)
    
    ks = [2, 4, 6, 8]
    inertias = []
    segs = []
    
    # 4. Perform K-Means clustering for each K value
    for k in ks:
        m = KMeans(n_clusters=k, n_init=10, random_state=42)
        lab = m.fit_predict(X)
        
        # Replace pixel values with cluster centroid colors
        seg = m.cluster_centers_[lab].reshape(arr.shape).astype('uint8')
        inertias.append(m.inertia_)
        segs.append(seg)
        
        # Save individual segmented output images
        Image.fromarray(seg).save(OUT / f'segmented_k{k}.png')
        
    # 5. Generate side-by-side comparison plot
    fig, ax = plt.subplots(1, 5, figsize=(18, 4))
    ax[0].imshow(arr)
    ax[0].set_title('Original')
    ax[0].axis('off')
    
    for a, k, s in zip(ax[1:], ks, segs):
        a.imshow(s)
        a.set_title(f'K={k}')
        a.axis('off')
        
    plt.tight_layout()
    plt.savefig(OUT / 'image_segmentation_comparison.png', dpi=150)
    plt.close()
    
    # 6. Generate Elbow curve plot
    plt.figure(figsize=(8, 5))
    plt.plot(ks, inertias, marker='o', color='#2b5c8f', linewidth=2)
    plt.xlabel('K (Number of Clusters)')
    plt.ylabel('Inertia')
    plt.title('Elbow Curve - Image Segmentation')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUT / 'image_segmentation_elbow.png', dpi=150)
    plt.close()
    
    print("\nSegmentation completed successfully!")
    print(f"All output files have been saved to: {OUT}")

if __name__ == '__main__':
    run()