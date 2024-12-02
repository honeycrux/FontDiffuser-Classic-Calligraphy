# Prepares reference dictionary

from src.build import build_style_encoder, build_content_encoder
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


def transform_image(args, image: Image.Image):
    style_inference_transforms = transforms.Compose(
        [transforms.Resize(args.style_image_size, \
                        interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])])
    image = style_inference_transforms(image)[None, :]
    return image

def use_style_encoder(args, style_encoder, image: torch.Tensor):
    encoded_style = style_encoder(image)
    return encoded_style

def use_content_encoder(args, content_encoder, image: torch.Tensor):
    encoded_style = content_encoder(image)
    return encoded_style

def fetch_lantingjixu_chars():
    with open('lantingjixu.txt', 'r', encoding='utf-8') as text_file:
        text = text_file.read()
        characters = list(set(text))
        return characters

def fetch_word_id_map():
    with open('wordlist.txt', 'r', encoding='utf-8') as f:
        line = f.read().strip()
        word_id_map = {word.strip()[0]: i for i, word in enumerate(line)}
        id_word_map = {i: word.strip()[0] for i, word in enumerate(line)}
    return word_id_map, id_word_map

def fetch_lantingjixu_char_files():
    word_id_map, _ = fetch_word_id_map()

    with open('lacklist.txt', 'r', encoding='utf-8') as f:
        line = f.read().strip()
        lack_list = list(line)

    # fetch lantingjixu characters but filter out the exception list
    characters = fetch_lantingjixu_chars()
    characters = [char for char in characters if char not in lack_list]

    character_files = [str(word_id_map[char]).rjust(5, '0') + '.png' for char in characters]
    return character_files


def build_reference_dict_s(args, style_encoder, selected_files = None):
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
            transformed_style_image = transform_image(args, style_image)
            style_img_feature, _, style_residual_features = use_style_encoder(args=args, style_encoder=style_encoder, image=transformed_style_image)
            encoded_references[file_name] = style_img_feature
    print(f"Number of references: {len(encoded_references)}")
    torch.save(encoded_references, f"{args.ckpt_dir}/encoded_references_s.pth")

    if selected_files is not None:
        for selected_file in selected_files:
            if selected_file not in encoded_references:
                print(f"Missing reference for {selected_file}")

def build_reference_dict_c(args, content_encoder, selected_files = None):
    def is_selected(file_name):
        if selected_files is None:
            return True
        return file_name in selected_files

    references_dir = Path(args.references_dir)
    file_names = []
    style_images = []
    for image_path in references_dir.iterdir():
        file_name = image_path.name
        if image_path.is_file() and is_selected(file_name):
            style_image = Image.open(image_path).convert('RGB')
            transformed_style_image = transform_image(args, style_image)
            file_names.append(file_name)
            style_images.append(transformed_style_image)
    style_images = torch.cat(style_images, dim=0)

    print(f"Number of references: {len(file_names)}")
    content_img_feature, content_residual_features = use_content_encoder(args=args, content_encoder=content_encoder, image=style_images)
    content_residual_features.append(content_img_feature)
    encoded_info = {
        "names": file_names,
        "encodings":content_residual_features,
    }
    torch.save(encoded_info, f"{args.ckpt_dir}/encoded_references_c.pth")

    if selected_files is not None:
        for selected_file in selected_files:
            if selected_file not in file_names:
                print(f"Missing reference for {selected_file}")

def reference_selection_s(args, style_encoder, encoded_references, content_image: torch.Tensor, loss_type='cosine'):
    style_img_feature, _, style_residual_features = use_style_encoder(args=args, style_encoder=style_encoder, image=content_image)

    is_similarity = True

    similarity = {}
    for image_id, encoded_reference in encoded_references.items():
        if loss_type == 'cosine':
            sim = torch.nn.functional.cosine_similarity(style_img_feature, encoded_reference) # returns 3x3 tensor when taking cosine similarity of two 1024x3x3 tensors
            similarity[image_id] = torch.sum(sim)
            is_similarity = True
        elif loss_type == 'mse':
            diff = torch.nn.functional.mse_loss(style_img_feature, encoded_reference)
            similarity[image_id] = diff
            is_similarity = False
    sorted_similarity = sorted(similarity.items(), key=lambda x: x[1], reverse=is_similarity)
    return sorted_similarity

def reference_selection_c(args, content_encoder, encoded_info, content_image: torch.Tensor, loss_type='cosine'):
    reference_file_names = encoded_info["names"]
    reference_encodings = encoded_info["encodings"]

    content_img_feature, content_residual_features = use_content_encoder(args=args, content_encoder=content_encoder, image=content_image)
    content_residual_features.append(content_img_feature)

    is_similarity = True

    similarity = {}
    for file_idx, file_name in enumerate(reference_file_names):
        if loss_type == 'cosine':
            total_sim = 0.0
            # for fs_idx in range(len(content_residual_features)):
                # item_sim = torch.nn.functional.cosine_similarity(content_residual_features[fs_idx], reference_encodings[fs_idx][file_idx].unsqueeze(dim=0))
                # total_sim += torch.sum(item_sim).item()
            fs_idx = -1
            item_sim = torch.nn.functional.cosine_similarity(content_residual_features[fs_idx], reference_encodings[fs_idx][file_idx].unsqueeze(dim=0))
            total_sim += torch.sum(item_sim).item()
            similarity[file_name] = total_sim
            is_similarity = True
        elif loss_type == 'mse':
            total_diff = 0.0
            # for fs_idx in range(len(content_residual_features)):
            #     item_diff = torch.nn.functional.mse_loss(content_residual_features[fs_idx], reference_encodings[fs_idx][file_idx].unsqueeze(dim=0))
            #     total_diff += item_diff.item()
            fs_idx = -1
            item_diff = torch.nn.functional.mse_loss(content_residual_features[fs_idx], reference_encodings[fs_idx][file_idx].unsqueeze(dim=0))
            total_diff += item_diff.item()
            similarity[file_name] = total_diff
            is_similarity = False
    sorted_similarity = sorted(similarity.items(), key=lambda x: x[1], reverse=is_similarity)
    return sorted_similarity

def build_reference_dict_example():
    args = arg_parse()
    args.ckpt_dir = 'ckpt/'
    args.references_dir = 'lantingxu_resized/id_0'

    # style_encoder = build_style_encoder(args=args)
    # style_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/style_encoder.pth"))
    content_encoder = build_content_encoder(args=args)
    content_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/content_encoder.pth"))

    selected_files = fetch_lantingjixu_char_files()
    print("Number of selected files: ", len(selected_files))

    # build_reference_dict_s(args=args, style_encoder=style_encoder, selected_files=selected_files)
    build_reference_dict_c(args=args, content_encoder=content_encoder, selected_files=selected_files)

def reference_selection_example(content_image_path):
    args = arg_parse()
    args.ckpt_dir = 'ckpt/'
    args.content_image_path = content_image_path

    content_image = Image.open(args.content_image_path).convert('RGB')
    transformed_content_image = transform_image(args, content_image)

    # style_encoder = build_style_encoder(args=args)
    # style_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/style_encoder.pth"))

    # encoded_references = torch.load(f"{args.ckpt_dir}/encoded_references_s.pth")
    # sorted_similarity = reference_selection_s(args=args,
    #                                           style_encoder=style_encoder,
    #                                           encoded_references=encoded_references,
    #                                           content_image=transformed_content_image,
    #                                           loss_type='cosine')

    content_encoder = build_content_encoder(args=args)
    content_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/content_encoder.pth"))

    encoded_info = torch.load(f"{args.ckpt_dir}/encoded_references_c.pth")
    sorted_similarity = reference_selection_c(args=args,
                                              content_encoder=content_encoder,
                                              encoded_info=encoded_info,
                                              content_image=transformed_content_image,
                                              loss_type='mse')

    return sorted_similarity


if __name__ == "__main__":
    build_reference_dict_example()

    # test reference selection
    # content_image_path = 'data_examples/sampling/example_content.jpg'
    # sims = reference_selection_example(content_image_path=content_image_path)
    # print(content_image_path)
    # print(sims[:5])

    # test reference selection version 2
    # _, id_word_map = fetch_word_id_map()
    # path_to_word = lambda w: id_word_map[int(w.replace(".png", ""))]
    # content_image_paths = ['02307.png', '02583.png', '00084.png', '01510.png']
    # root_dir = 'data/train/ContentImage/'
    # for path in content_image_paths:
    #     sims = reference_selection_example(content_image_path=root_dir+path)
    #     print(path)
    #     print(sims[:5])
    #     print(path_to_word(path))
    #     print(list(map(lambda x: path_to_word(x[0]), sims[:5])))