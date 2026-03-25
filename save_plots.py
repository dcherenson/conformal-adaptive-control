import os
import shutil
from datetime import datetime
import glob

def main():
    # Create an output directory with a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = f"run_{timestamp}"
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Find all png and gif files
    files = glob.glob("*.png") + glob.glob("*.gif")
    
    if not files:
        print("No .png or .gif files found in the current directory.")
        return
        
    # Move them to the new folder
    count = 0
    for file in files:
        target = os.path.join(out_dir, file)
        shutil.move(file, target)
        # To copy instead of move, use: shutil.copy2(file, target)
        print(f"Moved {file} -> {target}")
        count += 1
        
    print(f"\nSuccessfully moved {count} files to the directory: {out_dir}")

if __name__ == "__main__":
    main()
