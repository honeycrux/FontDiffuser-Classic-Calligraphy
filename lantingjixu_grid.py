import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from PIL import Image

def fetch_lantingjixu_text():
    with open('lantingjixu.txt', 'r', encoding='utf-8') as text_file:
        text = text_file.read()
        characters = list(text)
        return characters

def convert_to_grid(image_files, batch_size):
    # Split the image files into rows
    return [image_files[i:i + batch_size] for i in range(0, len(image_files), batch_size)]

# Function to create a grid of images
def display_images_in_grid(image_paths):
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
    plt.savefig('outputs/lantingjixu_grid.png')

if __name__ == '__main__':
    image_folder = 'outputs/'  # Set your image folder path
    image_files = []

    characters = fetch_lantingjixu_text()
    for character in characters:
        image_files.append(os.path.join(image_folder, f'{character}.png'))

    title_word_count = 4
    image_files_grid = [image_files[:title_word_count]] + convert_to_grid(image_files[title_word_count:], 13)

    display_images_in_grid(image_files_grid)
