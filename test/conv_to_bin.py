# Used ai for this file

import os
import pickle
import random
import itertools
import cv2
import numpy as np

def generate_pairs_and_bin(folder_root, output_bin_path, seed=1337):
    random.seed(seed)
    
    # 1. Collect all images group by person folder
    person_dirs = sorted([d for d in os.listdir(folder_root) if os.path.isdir(os.path.join(folder_root, d))])
    person_to_imgs = {}
    
    valid_exts = ('.jpg', '.jpeg', '.png')
    for p in person_dirs:
        imgs = [os.path.join(folder_root, p, f) for f in os.listdir(os.path.join(folder_root, p))
                if f.lower().endswith(valid_exts)]
        if len(imgs) >= 2:
            person_to_imgs[p] = sorted(imgs)

    if len(person_to_imgs) < 2:
        raise ValueError(f"Need at least 2 people with 2+ images in {folder_root} to generate pairs.")

    pos_pairs = []
    # 2. Build positive pairs (same person)
    for p, imgs in person_to_imgs.items():
        for i1, i2 in itertools.combinations(imgs, 2):
            pos_pairs.append((i1, i2, True))

    # 3. Build negative pairs (different people)
    neg_pairs = []
    person_keys = list(person_to_imgs.keys())
    for _ in range(len(pos_pairs)):
        p1, p2 = random.sample(person_keys, 2)
        im1 = random.choice(person_to_imgs[p1])
        im2 = random.choice(person_to_imgs[p2])
        neg_pairs.append((im1, im2, False))

    all_pairs = pos_pairs + neg_pairs
    random.shuffle(all_pairs)

    print(f"Creating .bin with {len(all_pairs)} total pairs ({len(pos_pairs)} same, {len(neg_pairs)} different)...")

    bins = []
    issame_list = []

    for img1_path, img2_path, issame in all_pairs:
        # Load and resize to 112x112
        im1 = cv2.imread(img1_path)
        im2 = cv2.imread(img2_path)
        
        im1 = cv2.resize(im1, (112, 112))
        im2 = cv2.resize(im2, (112, 112))

        # Encode to JPEG bytes
        _, enc1 = cv2.imencode('.jpg', im1)
        _, enc2 = cv2.imencode('.jpg', im2)

        bins.append(enc1.tobytes())
        bins.append(enc2.tobytes())
        issame_list.append(issame)

    # 4. Save pickle file
    with open(output_bin_path, 'wb') as f:
        pickle.dump((bins, np.array(issame_list)), f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved binary validation file to: {output_bin_path}")


if __name__ == '__main__':
    # Adjust paths as needed for your folder structure
    val_folder = "/home/devashish_tripathi/projects/sanity_data/data_bdct/val"
    out_bin = "/home/devashish_tripathi/projects/sanity_data/data_bdct/val_dummy.bin"
    generate_pairs_and_bin(val_folder, out_bin)