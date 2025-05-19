import os
import sys


def count_npy_files(folder_path):
    """
    Count the number of .npy files in the given folder and its subfolders.

    Args:
        folder_path (str): Path to the folder.

    Returns:
        int: Number of .npy files.
    """
    npy_count = 0
    for root, _, files in os.walk(folder_path):
        npy_count += sum(1 for file in files if file.endswith(".npy"))
    return npy_count


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python count_files.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        sys.exit(1)

    count = count_npy_files(folder_path)
    print(f"Number of .npy files in '{folder_path}': {count}")
