# Prepares reference dictionary

from src.build import build_style_encoder
from PIL import Image
from pathlib import Path

import torch
import torchvision.transforms as transforms

def arg_parse():
    from configs.fontdiffuser import get_parser

    parser = get_parser()

    # original
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--controlnet", type=bool, default=False, 
                        help="If in demo mode, the controlnet can be added.")
    parser.add_argument("--character_input", action="store_true")
    parser.add_argument("--content_character", type=str, default=None)
    parser.add_argument("--content_image_path", type=str, default=None)
    parser.add_argument("--style_image_path", type=str, default=None)
    parser.add_argument("--save_image", action="store_true")
    parser.add_argument("--save_image_dir", type=str, default=None,
                        help="The saving directory.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ttf_path", type=str, default="ttf/KaiXinSongA.ttf")

    # reference selection
    parser.add_argument("--references_dir", type=str, default=None)

    args = parser.parse_args()
    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)

    return args


def use_style_encoder(args, style_encoder, image: Image.Image):
    style_inference_transforms = transforms.Compose(
        [transforms.Resize(args.style_image_size, \
                        interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])])
    image = style_inference_transforms(image)[None, :]
    encoded_style = style_encoder(image)
    return encoded_style

def fetch_lantingjixu_chars():
    with open('lantingjixu.txt', 'r', encoding='utf-8') as text_file:
        text = text_file.read()
        characters = list(set(text))
        return characters

def fetch_lantingjixu_char_files():
    with open('strokelist.txt', 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
        word_id_map = {line.strip()[0]: i for i, line in enumerate(lines)}

    with open('lacklist.txt', 'r', encoding='utf-8') as f:
        line = f.read().strip()
        lack_list = list(line)

    # fetch lantingjixu characters but filter out the exception list
    characters = fetch_lantingjixu_chars()
    characters = [char for char in characters if char not in lack_list]

    character_files = [str(word_id_map[char]).rjust(5, '0') + '.png' for char in characters]
    return character_files


def build_reference_dict(args, style_encoder, selected_files = None):
    def is_selected(file_name):
        if selected_files is None:
            return True
        return file_name in selected_files

    references_dir = Path(args.references_dir)
    encoded_references = {}
    for image_path in references_dir.iterdir():
        file_name = image_path.name
        if image_path.is_file() and is_selected(file_name):
            style_image = Image.open(image_path).convert('RGB')
            style_img_feature, _, style_residual_features = use_style_encoder(args=args, style_encoder=style_encoder, image=style_image)
            encoded_references[file_name] = style_img_feature
    print(f"Number of references: {len(encoded_references)}")
    torch.save(encoded_references, f"{args.ckpt_dir}/encoded_references.pth")

    if selected_files is not None:
        for selected_file in selected_files:
            if selected_file not in encoded_references:
                print(f"Missing reference for {selected_file}")

def reference_selection(args, style_encoder, encoded_references, content_image: Image.Image):
    content_img_feature, _, content_residual_features = use_style_encoder(args=args, style_encoder=style_encoder, image=content_image)

    loss_type = 'mse'

    similarity = {}
    for image_id, encoded_reference in encoded_references.items():
        if loss_type == 'cosine':
            similarity_item = torch.nn.functional.cosine_similarity(content_img_feature, encoded_reference) # returns 3x3 tensor when taking cosine similarity of two 1024x3x3 tensors
            similarity[image_id] = torch.sum(similarity_item)
        elif loss_type == 'mse':
            similarity_item = torch.nn.functional.mse_loss(content_img_feature, encoded_reference)
            similarity[image_id] = similarity_item
    sorted_similarity = sorted(similarity.items(), key=lambda x: x[1], reverse=True)
    return sorted_similarity


def build_reference_dict_example():
    args = arg_parse()
    args.ckpt_dir = 'ckpt/'
    args.references_dir = 'lantingxu_resized/id_0'

    style_encoder = build_style_encoder(args=args)
    style_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/style_encoder.pth"))

    selected_files = fetch_lantingjixu_char_files()
    print("Number of selected files: ", len(selected_files))

    build_reference_dict(args=args, style_encoder=style_encoder, selected_files=selected_files)

def reference_selection_example():
    args = arg_parse()
    args.ckpt_dir = 'ckpt/'
    args.content_image_path = 'data_examples/sampling/example_content.jpg'

    style_encoder = build_style_encoder(args=args)
    style_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/style_encoder.pth"))

    encoded_references = torch.load(f"{args.ckpt_dir}/encoded_references.pth")

    content_image = Image.open(args.content_image_path).convert('RGB')

    sorted_similarity = reference_selection(args=args, style_encoder=style_encoder, encoded_references=encoded_references, content_image=content_image)
    return sorted_similarity


if __name__ == "__main__":
    build_reference_dict_example()

    # test reference selection
    # sims = reference_selection_example()
    # print(sims[:5])
