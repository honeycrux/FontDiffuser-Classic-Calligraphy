# Prepares reference dictionary

from src.build import build_style_encoder
from PIL import Image
from pathlib import Path

import torch
import torchvision.transforms as transforms

def arg_parse():
    from configs.fontdiffuser import get_parser

    parser = get_parser()
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
    args = parser.parse_args()
    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)

    return args

def use_style_encoder(style_encoder, style_image: Image.Image):
    style_inference_transforms = transforms.Compose(
        [transforms.Resize(args.style_image_size, \
                        interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])])
    style_image = style_inference_transforms(style_image)[None, :]
    encoded_style = style_encoder(style_image)
    return encoded_style


def build_reference_dict(args, style_encoder):
    references_dir = Path(args.references_dir)
    encoded_references = {}
    for image_path in references_dir.iterdir():
        image_id = image_path.stem
        if image_path.is_file():
            style_image = Image.open(image_path).convert('RGB')
            style_img_feature, _, style_residual_features = use_style_encoder(style_encoder, style_image)
            encoded_references[image_id] = style_img_feature
    torch.save(encoded_references, f"{args.ckpt_dir}/encoded_references.pth")

def reference_selection(args, style_encoder, encoded_references, content_image: Image.Image):
    content_img_feature, _, content_residual_features = use_style_encoder(style_encoder, content_image)
    print(content_img_feature.shape)

    similarity = {}
    for image_id, encoded_reference in encoded_references.items():
        similarity_item = torch.nn.functional.cosine_similarity(content_img_feature, encoded_reference)
        print(similarity_item)
        similarity[image_id] = torch.sum(similarity_item)
    sorted_similarity = sorted(similarity.items(), key=lambda x: x[1], reverse=True)
    return sorted_similarity

def call_build_reference_dict(args):
    style_encoder = build_style_encoder(args=args)
    style_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/style_encoder.pth"))

    build_reference_dict(args, style_encoder)

def call_reference_selection(args):
    style_encoder = build_style_encoder(args=args)
    style_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/style_encoder.pth"))

    encoded_references = torch.load(f"{args.ckpt_dir}/encoded_references.pth")

    content_image = Image.open(args.content_image_path).convert('RGB')

    sorted_similarity = reference_selection(args, style_encoder, encoded_references, content_image)
    return sorted_similarity

if __name__ == "__main__":
    args = arg_parse()
    args.ckpt_dir = 'ckpt/'
    args.references_dir = 'data_examples/references'
    args.content_image_path = 'data_examples/sampling/example_content.jpg'

    encoded_references = torch.load(f"{args.ckpt_dir}/encoded_references.pth")
    print(call_reference_selection(args))
