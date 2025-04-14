# This script is provided by the FYP24 project group.
# This is the driver code for creating a grid of characters (top-to-bottom, right-to-left).
# It is configured to use the LantingjiXu text and the images generated from the LantingjiXu text.
# The LantingjiXu text must first be generated using lantingjixu_sample.py.
# The image path, save path, and text data path can be configured in the main function.

import matplotlib.pyplot as plt
import os
from PIL import Image

def load_text(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as text_file:
        text = text_file.read()
        characters = list(text)
        return characters

def convert_to_grid(image_files: list[str], line_size: int):
    # Split the image files into rows
    return [image_files[i:i + line_size] for i in range(0, len(image_files), line_size)]

# Function to create a grid of images
def display_images_in_grid(image_paths: list[list[str]], save_location: str):
    # Calculate the number of rows and cols needed
    cols = len(image_paths)
    rows = max([len(image_line) for image_line in image_paths])
    print(cols, rows)
    
    # Create a new figure
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15), gridspec_kw={'wspace': 0, 'hspace': 0}, layout='compressed')

    for line_no, image_line in enumerate(image_paths):
        for word_no, path in enumerate(image_line):
            img = Image.open(path)
            col_no = len(image_paths) - line_no - 1
            row_no = word_no
            axes[row_no, col_no].imshow(img)
            axes[row_no, col_no].axis('off')

    # Hide any remaining empty subplots
    for ax_row in axes:
        for ax in ax_row:
            ax.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')

    # plt.tight_layout(pad=0)
    # plt.subplots_adjust(wspace=0, hspace=0)
    # plt.show()
    plt.savefig(save_location)

def main():
    image_folder = 'outputs/'  # Set image folder path with the generated images

    save_path = 'outputs/lantingjixu_grid.png' # Set the location to save the grid

    line_size = 13 # Set the number of characters on each vertical line

    require_title = True # Whether to include the title

    title_data_path = 'data_lantingjixu/lantingjixu_title.txt' # Set the path to the title
    text_data_path = 'data_lantingjixu/lantingjixu_used.txt' # Set the path to the text

    text_characters = load_text(text_data_path)
    text_image_files: list[str] = []
    for character in text_characters:
        text_image_files.append(os.path.join(image_folder, f'{character}.png'))

    text_image_files_grid = convert_to_grid(text_image_files, line_size)

    image_files_grid = text_image_files_grid

    if require_title:
        title_characters = load_text(title_data_path)
        title_image_files: list[str] = []
        for character in title_characters:
            title_image_files.append(os.path.join(image_folder, f'{character}.png'))

        title_image_files_grid = convert_to_grid(title_image_files, line_size)
        empty_line = []

        image_files_grid = title_image_files_grid + [empty_line] + text_image_files_grid

    display_images_in_grid(image_files_grid, save_path)

if __name__ == '__main__':
    main()
