# This script is provided by the FYP24 project group.

import torch
from torch import nn

from .style_absorption_blocks import DecoderBlock, EncoderBlock, Transformer


def verify_content_encodings(content_encodings: list[torch.Tensor]) -> None:
    assert len(content_encodings) == 5, "There should be 5 content encodings."
    expected_shapes = [
        (3, 96, 96),
        (64, 48, 48),
        (128, 24, 24),
        (256, 12, 12),
        (256, 12, 12),
    ]
    for i, encoding in enumerate(content_encodings):
        if encoding.dim() == 5:
            B, K, C, H, W = encoding.shape
        else:
            B, C, H, W = encoding.shape
            K = 1
        expected_C, expected_H, expected_W = expected_shapes[i]
        assert (
            C == expected_C and H == expected_H and W == expected_W
        ), f"Content encoding {i} has incorrect shape. Expected ({expected_C}, {expected_H}, {expected_W}), got ({C}, {H}, {W})."


def verify_style_encoding(style_encoding: torch.Tensor) -> None:
    if style_encoding.dim() == 5:
        B, K, C, H, W = style_encoding.shape
    else:
        B, C, H, W = style_encoding.shape
        K = 1
    expected_C, expected_H, expected_W = (1024, 3, 3)
    assert (
        C == expected_C and H == expected_H and W == expected_W
    ), f"Style encoding has incorrect shape. Expected ({expected_C}, {expected_H}, {expected_W}), got ({C}, {H}, {W})."


class ReferenceImageEncodings:
    def __init__(
        self,
        computer_font_content_encodings: list[
            torch.Tensor
        ],  # 5 encodings of (B, K, C, H, W)
        actual_content_encodings: list[torch.Tensor],  # 5 encodings of (B, K, C, H, W)
        actual_style_encoding: torch.Tensor,  # (B, K, C, H, W)
    ):
        verify_content_encodings(computer_font_content_encodings)
        verify_content_encodings(actual_content_encodings)
        verify_style_encoding(actual_style_encoding)
        self.computer_font_content_encodings = computer_font_content_encodings
        self.actual_content_encodings = actual_content_encodings
        self.actual_style_encoding = actual_style_encoding


class SourceImageEncodings:
    def __init__(
        self,
        computer_font_content_encodings: list[
            torch.Tensor
        ],  # 5 encodings of (B, C, H, W)
        neutral_style_encoding: torch.Tensor,  # (B, C, H, W)
    ):
        verify_content_encodings(computer_font_content_encodings)
        verify_style_encoding(neutral_style_encoding)
        self.computer_font_content_encodings = computer_font_content_encodings
        self.neutral_style_encoding = neutral_style_encoding


class OutputImageEncodings:
    def __init__(
        self,
        content_encodings: list[torch.Tensor],
        style_encoding: torch.Tensor,
    ):
        verify_content_encodings(content_encodings)
        verify_style_encoding(style_encoding)
        self.content_encodings = content_encodings
        self.style_encoding = style_encoding


class EncodingShape:
    actual_shape: tuple[int, int, int]  # (C, H, W)
    projected_shape: tuple[int, int, int]  # (C, H, W)

    def __init__(
        self, actual_shape: tuple[int, int, int], projected_shape: tuple[int, int, int]
    ):
        self.actual_shape = actual_shape
        self.projected_shape = projected_shape


class StyleAbsorption(nn.Module):
    """
    Transformer for multiple reference images (in multi-scale content encoding of the computer font image,
    multi-scale content encoding of the reference image, and style encoding of the reference image)
    and a single content image (in multi-scale content encoding of the computer font image and a neutral style encoding).
    The multi-scale content encodings are 5 encodings of different resolutions.

    First, for each encoding, project the input (aka embedding) and reshape to b, t, d where d=1024.
    Concatenate all 11 encodings of the reference images as the encoder input.
    Concatenate all 5 encodings and the neutral style encoding of the content image as the decoder input.
    Apply standard transformer action.
    Finally, reshape the output back to the original encoding shapes.

    Parameters:
        n_heads (:obj:`int`): The number of heads to use for multi-head attention.
        d_head (:obj:`int`): The number of channels in each head.
        n_layers (:obj:`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (:obj:`float`, *optional*, defaults to 0.1): The dropout probability to use.
        context_dim (:obj:`int`, *optional*): The number of context dimensions to use.
    """

    def __init__(
        self,
        n_heads: int,
        d_head: int,
        decoder_query_dim: int,
        encoder_query_dim: int,
        n_layers: int = 1,
        dropout: float = 0.0,
        gated_ff=True,
        ff_mult: int = 4,
    ):
        super().__init__()

        self.encoding_shapes = [
            EncodingShape(actual_shape=(3, 96, 96), projected_shape=(1, 32, 32)),
            EncodingShape(actual_shape=(64, 48, 48), projected_shape=(4, 16, 16)),
            EncodingShape(actual_shape=(128, 24, 24), projected_shape=(16, 8, 8)),
            EncodingShape(actual_shape=(256, 12, 12), projected_shape=(64, 4, 4)),
            EncodingShape(actual_shape=(256, 12, 12), projected_shape=(64, 4, 4)),
            EncodingShape(actual_shape=(1024, 3, 3), projected_shape=(1024, 1, 1)),
        ]

        self.proj_in = nn.ModuleList(
            [
                nn.Conv2d(
                    3, 1, kernel_size=3, stride=3, padding=0
                ),  # (B, 3, 96, 96) -> (B, 1, 32, 32)
                nn.Conv2d(
                    64, 4, kernel_size=3, stride=3, padding=0
                ),  # (B, 64, 48, 48) -> (B, 4, 16, 16)
                nn.Conv2d(
                    128, 16, kernel_size=3, stride=3, padding=0
                ),  # (B, 128, 24, 24) -> (B, 16, 8, 8)
                nn.Conv2d(
                    256, 64, kernel_size=3, stride=3, padding=0
                ),  # (B, 256, 12, 12) -> (B, 64, 4, 4)
                nn.Conv2d(
                    256, 64, kernel_size=3, stride=3, padding=0
                ),  # (B, 256, 12, 12) -> (B, 64, 4, 4)
                nn.Conv2d(
                    1024, 1024, kernel_size=3, stride=3, padding=0
                ),  # (B, 1024, 3, 3) -> (B, 1024, 1, 1)
            ]
        )

        encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    query_dim=encoder_query_dim,
                    n_heads=n_heads,
                    d_head=d_head,
                    dropout=dropout,
                    gated_ff=gated_ff,
                    ff_mult=ff_mult,
                )
                for _ in range(n_layers)
            ]
        )

        decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    query_dim=decoder_query_dim,
                    n_heads=n_heads,
                    d_head=d_head,
                    dropout=dropout,
                    context_dim=encoder_query_dim,
                    gated_ff=gated_ff,
                    ff_mult=ff_mult,
                )
                for _ in range(n_layers)
            ]
        )

        self.transformer = Transformer(
            encoder_blocks=encoder_blocks, decoder_blocks=decoder_blocks
        )

        self.proj_out = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    1, 3, kernel_size=3, stride=3, padding=0
                ),  # (B, 1, 32, 32) -> (B, 3, 96, 96)
                nn.ConvTranspose2d(
                    4, 64, kernel_size=3, stride=3, padding=0
                ),  # (B, 4, 16, 16) -> (B, 64, 48, 48)
                nn.ConvTranspose2d(
                    16, 128, kernel_size=3, stride=3, padding=0
                ),  # (B, 16, 8, 8) -> (B, 128, 24, 24)
                nn.ConvTranspose2d(
                    64, 256, kernel_size=3, stride=3, padding=0
                ),  # (B, 64, 4, 4) -> (B, 256, 12, 12)
                nn.ConvTranspose2d(
                    64, 256, kernel_size=3, stride=3, padding=0
                ),  # (B, 64, 4, 4) -> (B, 256, 12, 12)
                nn.ConvTranspose2d(
                    1024, 1024, kernel_size=3, stride=3, padding=0
                ),  # (B, 1024, 1, 1) -> (B, 1024, 3, 3)
            ]
        )

    def project_encoding(self, encoding: torch.Tensor, projector: nn.Module):
        if encoding.dim() == 5:
            B, K, C, H, W = encoding.shape
        else:
            B, C, H, W = encoding.shape
            K = 1
        proj_encoding = encoding.view(B * K, C, H, W)
        proj_encoding = projector(proj_encoding)  # (B*K, C1, H1, W1)
        BK, C1, H1, W1 = proj_encoding.shape
        assert BK == B * K, "Batch size mismatch after projection ({BK} != {B} * {K})"
        assert (
            C1 * H1 * W1 == 1024
        ), f"Projected dimensions ({C1}, {H1}, {W1}) does not multiply to 1024"
        proj_encoding = proj_encoding.permute(0, 2, 3, 1)  # (B*K, H1, W1, C1)
        proj_encoding = proj_encoding.reshape(
            B, K, 1024
        )  # (B, T = K, D = C1*H1*W1 = 1024)
        return proj_encoding

    def reshape_to_encoding(
        self,
        transformer_output: torch.Tensor,
        projector: nn.Module,
        projected_shape: tuple[int, int, int],
    ):
        B, T, D = transformer_output.shape
        assert T == 1, f"Transformer output sequence length {T} is not 1"
        assert D == 1024, f"Transformer output dimension {D} is not 1024"
        C1, H1, W1 = projected_shape
        output_encoding = transformer_output.view(B, H1, W1, C1).permute(
            0, 3, 1, 2
        )  # (B, C1, H1, W1)
        output_encoding = projector(output_encoding)  # (B, C, H, W)
        return output_encoding

    def forward(
        self,
        reference_image_encodings: ReferenceImageEncodings,
        source_image_encodings: SourceImageEncodings,
    ):
        ref_computer_font_content_encodings = []
        for i, encoding in enumerate(
            reference_image_encodings.computer_font_content_encodings
        ):
            proj_encoding = self.project_encoding(
                encoding, self.proj_in[i]
            )  # (B, T, D=1024)
            ref_computer_font_content_encodings.append(proj_encoding)

        ref_actual_content_encodings = []
        for i, encoding in enumerate(
            reference_image_encodings.actual_content_encodings
        ):
            proj_encoding = self.project_encoding(
                encoding, self.proj_in[i]
            )  # (B, T, D=1024)
            ref_actual_content_encodings.append(proj_encoding)

        ref_actual_style_encoding = reference_image_encodings.actual_style_encoding
        ref_actual_style_encoding = self.project_encoding(
            ref_actual_style_encoding, self.proj_in[-1]
        )  # (B, T, D=1024)

        encoder_input = torch.cat(
            ref_computer_font_content_encodings
            + ref_actual_content_encodings
            + [ref_actual_style_encoding],
            dim=2,
        )  # (B, T, D_total = 1024 * 11)

        source_computer_font_content_encodings = []
        for i, encoding in enumerate(
            source_image_encodings.computer_font_content_encodings
        ):
            proj_encoding = self.project_encoding(
                encoding=encoding, projector=self.proj_in[i]
            )  # (B, T, D=1024)
            source_computer_font_content_encodings.append(proj_encoding)

        source_neutral_style_encoding = source_image_encodings.neutral_style_encoding
        source_neutral_style_encoding = self.project_encoding(
            encoding=source_neutral_style_encoding, projector=self.proj_in[-1]
        )  # (B, T, D=1024)

        decoder_input = torch.cat(
            source_computer_font_content_encodings + [source_neutral_style_encoding],
            dim=2,
        )  # (B, T, D_total = 1024 * 6)

        transformer_output = self.transformer(
            decoder_input=decoder_input, encoder_input=encoder_input
        )  # (B, T, D_total = 1024 * 6)

        output_content_encodings = []
        for i in range(5):
            output_encoding = self.reshape_to_encoding(
                transformer_output=transformer_output[:, :, i * 1024 : (i + 1) * 1024],
                projector=self.proj_out[i],
                projected_shape=self.encoding_shapes[i].projected_shape,
            )  # (B, C, H, W)
            output_content_encodings.append(output_encoding)

        output_style_encoding = self.reshape_to_encoding(
            transformer_output=transformer_output[:, :, 5 * 1024 : 6 * 1024],
            projector=self.proj_out[-1],
            projected_shape=self.encoding_shapes[-1].projected_shape,
        )  # (B, C, H, W)

        return OutputImageEncodings(
            content_encodings=output_content_encodings,
            style_encoding=output_style_encoding,
        )


if __name__ == "__main__":
    model = StyleAbsorption(
        n_heads=8,
        d_head=128,
        decoder_query_dim=1024 * 6,
        encoder_query_dim=1024 * 11,
        n_layers=1,
        gated_ff=True,
        ff_mult=1,
    )

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, f"{param.numel():,}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters in StyleAbsorption: {num_params:,}")
