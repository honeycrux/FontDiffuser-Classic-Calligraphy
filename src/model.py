# This script is provided by authors of FontDiffuser.

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from src.modules.style_absorption import (
    OutputImageEncodings,
    ReferenceImageEncodings,
    SourceImageEncodings,
)


class FontDiffuserModel(ModelMixin, ConfigMixin):
    """Forward function for FontDiffuser with content encoder \
        style encoder and unet.
    """

    @register_to_config
    def __init__(
        self,
        unet,
        style_encoder,
        content_encoder,
        style_absorption,
    ):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder
        self.style_absorption = style_absorption

    def forward(
        self,
        x_t,
        timesteps,
        style_images_batch,
        style_images_in_computer_font_batch,
        content_image_batch,
        content_encoder_downsample_size,
    ):
        # Part I: Get style and content features from style and content images

        ### Get style feature from style image *list*
        # style_batch are in the shape of (N, K, C, H, W)
        style_style_feature_list = []
        for style_images in style_images_batch:
            style_style_feature, _, _ = self.config["style_encoder"](style_images)
            style_style_feature_list.append(style_style_feature)
        style_style_feature_batch = torch.stack(style_style_feature_list)

        ### Get content feature from content image
        content_content_feture, content_content_residual_features = self.config[
            "content_encoder"
        ](content_image_batch)
        content_content_residual_features.append(content_content_feture)

        ### Get content feature from style image *list*
        style_content_residual_features_batch_transpose = []
        for style_images in style_images_batch:
            style_content_feature, style_content_residual_features = self.config[
                "content_encoder"
            ](style_images)
            style_content_residual_features.append(style_content_feature)
            style_content_residual_features_batch_transpose.append(
                style_content_residual_features
            )
        style_content_residual_features_batch = []
        for fs_idx in range(len(style_content_residual_features_batch_transpose[0])):
            # stack Fs_i columns
            Fs_i = [
                Ic_i[fs_idx] for Ic_i in style_content_residual_features_batch_transpose
            ]
            style_content_residual_features_batch.append(torch.stack(Fs_i))

        ### Get computer font content feature of style image *list*
        style_computer_font_content_residual_features_batch_transpose = []
        for style_images in style_images_in_computer_font_batch:
            (
                style_computer_font_content_feature,
                style_computer_font_content_residual_features,
            ) = self.config["content_encoder"](style_images)
            style_computer_font_content_residual_features.append(
                style_computer_font_content_feature
            )
            style_computer_font_content_residual_features_batch_transpose.append(
                style_computer_font_content_residual_features
            )
        style_computer_font_content_residual_features_batch = []
        for fs_idx in range(
            len(style_computer_font_content_residual_features_batch_transpose[0])
        ):
            # stack Fs_i columns
            Fs_i = [
                Ic_i[fs_idx]
                for Ic_i in style_computer_font_content_residual_features_batch_transpose
            ]
            style_computer_font_content_residual_features_batch.append(
                torch.stack(Fs_i)
            )

        ### Get neutral style encoding
        neutral_style_encoding, _, _ = self.config["style_encoder"](
            torch.ones_like(content_image_batch).to(self.device)
        )

        # Part II: infer *one* style_style_feature from K of them
        # and infer *one* style_content_residual_features from K of them

        output_encodings: OutputImageEncodings = self.config["style_absorption"](
            ReferenceImageEncodings(
                computer_font_content_encodings=style_computer_font_content_residual_features_batch,
                actual_content_encodings=style_content_residual_features_batch,
                actual_style_encoding=style_style_feature_batch,
            ),
            SourceImageEncodings(
                computer_font_content_encodings=content_content_residual_features,
                neutral_style_encoding=neutral_style_encoding,
            ),
        )
        style_style_feature = output_encodings.style_encoding
        style_content_residual_features = output_encodings.content_encodings

        # Part III: Do the rest and run the UNet

        batch_size, channel, height, width = style_style_feature.shape
        style_hidden_states = style_style_feature.permute(0, 2, 3, 1).reshape(
            batch_size, height * width, channel
        )

        input_hidden_states = [
            style_style_feature,
            content_content_residual_features,
            style_hidden_states,
            style_content_residual_features,
        ]

        out = self.config["unet"](
            x_t,
            timesteps,
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        noise_pred = out[0]
        offset_out_sum = out[1]

        return noise_pred, offset_out_sum


class FontDiffuserModelDPM(ModelMixin, ConfigMixin):
    """DPM Forward function for FontDiffuser with content encoder \
        style encoder and unet.
    """

    @register_to_config
    def __init__(
        self,
        unet,
        style_encoder,
        content_encoder,
        style_absorption,
    ):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder
        self.style_absorption = style_absorption

    def forward(
        self,
        x_t,
        timesteps,
        cond,
        content_encoder_downsample_size,
        version,
    ):
        content_image_batch = cond[0]
        style_images_batch = cond[1]
        style_images_in_computer_font_batch = cond[2]

        # Part I: Get style and content features from style and content images

        ### Initialization
        K = len(style_images_batch) // 2
        uncond_style_batch = style_images_batch[0:K]
        cond_style_batch = style_images_batch[K:]
        uncond_style_computer_font_batch = style_images_in_computer_font_batch[0:K]
        cond_style_computer_font_batch = style_images_in_computer_font_batch[K:]

        ### Get style feature from style image *list*
        uncond_style_style_feature, _, _ = self.config["style_encoder"](
            uncond_style_batch
        )
        cond_style_style_feature, _, _ = self.config["style_encoder"](cond_style_batch)
        combined_style_style_feature = torch.stack(
            [uncond_style_style_feature, cond_style_style_feature]
        )

        ### Get content feature from content image
        content_content_feture, content_content_residual_features = self.config[
            "content_encoder"
        ](content_image_batch)
        content_content_residual_features.append(content_content_feture)

        ### Get content feature from style image *list*
        uncond_style_content_feature, uncond_style_content_residual_features = (
            self.config["content_encoder"](uncond_style_batch)
        )
        uncond_style_content_residual_features.append(uncond_style_content_feature)
        cond_style_content_feature, cond_style_content_residual_features = self.config[
            "content_encoder"
        ](cond_style_batch)
        cond_style_content_residual_features.append(cond_style_content_feature)
        combined_style_content_residual_features = [
            torch.stack(
                [
                    uncond_style_content_residual_features[i],
                    cond_style_content_residual_features[i],
                ]
            )
            for i in range(len(uncond_style_content_residual_features))
        ]

        ### Get computer font content feature of style image *list*
        (
            uncond_style_computer_font_feature,
            uncond_style_computer_font_residual_features,
        ) = self.config["content_encoder"](uncond_style_computer_font_batch)
        uncond_style_computer_font_residual_features.append(
            uncond_style_computer_font_feature
        )
        (
            cond_style_computer_font_feature,
            cond_style_computer_font_residual_features,
        ) = self.config["content_encoder"](cond_style_computer_font_batch)
        cond_style_computer_font_residual_features.append(
            cond_style_computer_font_feature
        )
        combined_style_computer_font_residual_features = [
            torch.stack(
                [
                    uncond_style_computer_font_residual_features[i],
                    cond_style_computer_font_residual_features[i],
                ]
            )
            for i in range(len(uncond_style_computer_font_residual_features))
        ]

        ### Get neutral style encoding
        neutral_style_encoding, _, _ = self.config["style_encoder"](
            torch.ones_like(content_image_batch).to(self.device)
        )

        # Part II: infer *one* style_style_feature from K of them
        # and infer *one* style_content_residual_features from K of them

        output_encodings: OutputImageEncodings = self.config["style_absorption"](
            ReferenceImageEncodings(
                computer_font_content_encodings=combined_style_computer_font_residual_features,
                actual_content_encodings=combined_style_content_residual_features,
                actual_style_encoding=combined_style_style_feature,
            ),
            SourceImageEncodings(
                computer_font_content_encodings=content_content_residual_features,
                neutral_style_encoding=neutral_style_encoding,
            ),
        )
        style_style_feature = output_encodings.style_encoding
        style_content_residual_features = output_encodings.content_encodings

        # Part III: Do the rest and run the UNet

        batch_size, channel, height, width = style_style_feature.shape
        style_hidden_states = style_style_feature.permute(0, 2, 3, 1).reshape(
            batch_size, height * width, channel
        )

        input_hidden_states = [
            style_style_feature,
            content_content_residual_features,
            style_hidden_states,
            style_content_residual_features,
        ]

        out = self.config["unet"](
            x_t,
            timesteps,
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        noise_pred = out[0]

        return noise_pred
