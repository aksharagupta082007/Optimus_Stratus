import os
import glob
import random
import shutil

def main():
    src_dir = r"C:\Users\devar\OneDrive\Desktop\Projects\PBL-cubesat\dataset"
    dest_dir = r"C:\Users\devar\OneDrive\Desktop\Projects\PBL-cubesat\Optimus_Stratus\Optimus_Stratus\data\tfrecords"
    
    # Create dest dirs
    train_dir = os.path.join(dest_dir, "train")
    val_dir = os.path.join(dest_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Get all tfrecords
    all_tfrecords = glob.glob(os.path.join(src_dir, "**", "*.tfrecord"), recursive=True)
    print(f"Found {len(all_tfrecords)} TFRecord files.")
    
    random.seed(42)
    random.shuffle(all_tfrecords)
    
    split_idx = int(len(all_tfrecords) * 0.8)
    train_files = all_tfrecords[:split_idx]
    val_files = all_tfrecords[split_idx:]
    
    print(f"Assigning {len(train_files)} to train, {len(val_files)} to val.")
    
    for f in train_files:
        shutil.copy(f, os.path.join(train_dir, os.path.basename(f)))
        
    for f in val_files:
        shutil.copy(f, os.path.join(val_dir, os.path.basename(f)))
        
    print("Done preparing dataset.")

if __name__ == "__main__":
    main()
