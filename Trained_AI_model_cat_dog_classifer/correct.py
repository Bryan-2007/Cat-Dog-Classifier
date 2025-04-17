import os
from PIL import Image

def remove_corrupted_images(directory):
    num_removed = 0
    for subdir in os.listdir(directory):
        folder_path = os.path.join(directory, subdir)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Check if it's a real image
            except (IOError, SyntaxError):
                print(f"Removing corrupted file: {file_path}")
                os.remove(file_path)
                num_removed += 1
    print(f"\nDone. Removed {num_removed} corrupted images.")

# Set this to your actual training directory path
remove_corrupted_images("C:/Users/LENOVO/Desktop/DOGs VS CATs/data/train")  # e.g., "data/train"
remove_corrupted_images("C:/Users/LENOVO/Desktop/DOGs VS CATs/data/val")  # if needed
