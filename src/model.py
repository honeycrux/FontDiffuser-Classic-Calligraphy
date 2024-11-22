import math
import torch
import torch.nn as nn

from diffusers import ModelMixin
from diffusers.configuration_utils import (ConfigMixin, 
                                           register_to_config)
from src.modules.k_feature_extractor import KFeatureExtractor

class FontDiffuserModel(ModelMixin, ConfigMixin):           #FontDiffuserModel is a class which is inherited from ModelMixin and ConfigMixin
    """Forward function for FontDiffuer with content encoder \
        style encoder and unet.
    """

    @register_to_config
    def __init__(
        self, 
        unet, 
        style_encoder,
        content_encoder,
        k_feature_extractor,
    ):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder
        self.k_feature_extractor = k_feature_extractor

    def forward(
        self, 
        x_t, 
        timesteps, 
        style_images,
        content_images,
        content_encoder_downsample_size,
    ):
        # Part I: Get style and content features from style and content images

        ## Original implementation: one style image
        ### Get style feature from style image
        # style_style_feature, _, _ = self.style_encoder(style_images)

        ### Get content feature from content image
        # content_content_feature, content_content_residual_features = self.content_encoder(content_images)
        # content_content_residual_features.append(content_content_feature)

        ### Get content feature from style image
        # style_content_feature, style_content_residual_features = self.content_encoder(style_images)
        # style_content_residual_features.append(style_content_feature)

        ## Implementation 1: take average of the K style & content features from the style images

        ### Get style feature from style image *list*
        # style_images are in the shape of (N, K, C, H, W)
        # style_style_feature_list=[]
        # for k in range(style_images.size(1)):
        #     style_image_batch = style_images[:, k, :, :, :]
        #     style_style_feature, _, style_style_residual_features = self.style_encoder(style_image_batch)
        #     style_style_feature_list.append(style_style_feature)

        ### Get content feature from content image
        # content_content_feture, content_content_residual_features = self.content_encoder(content_images)
        # content_content_residual_features.append(content_content_feture)

        ### Get content feature from style image *list*
        # style_content_residual_features_list=[]
        # for k in range(style_images):
        #     style_image_batch = style_images[:, k, :, :, :]
        #     style_content_feature, style_content_residual_features = self.content_encoder(style_image_batch)
        #     style_content_residual_features.append(style_content_feature)
        #     style_content_residual_features_list.append(style_content_residual_features)

        ## Implementation 2: take learned feature of the K style & content features from the style images

        ### Initialization
        style_batch = style_images

        ### Get style feature from style image *list*
        # style_batch are in the shape of (N, K, C, H, W)
        style_style_feature_list = []
        for style_batch_item in style_batch:
            style_style_feature, _, _ = self.config.style_encoder(style_batch_item)
            style_style_feature_list.append(style_style_feature)
        style_style_feature_batch = torch.stack(style_style_feature_list)

        ### Get content feature from content image
        content_content_feture, content_content_residual_features = self.config.content_encoder(content_images)
        content_content_residual_features.append(content_content_feture)

        ### Get content feature from style image *list*
        style_content_residual_features_batch_transpose = []
        for style_batch_item in style_batch:
            style_content_feature, style_content_residual_features = self.config.content_encoder(style_batch_item)
            style_content_residual_features.append(style_content_feature)
            style_content_residual_features_batch_transpose.append(style_content_residual_features)
        style_content_residual_features_batch = []
        for fs_idx in range(len(style_content_residual_features_batch_transpose[0])):
            # stack Fs_i columns
            Fs_i = [Ic_i[fs_idx] for Ic_i in style_content_residual_features_batch_transpose]
            style_content_residual_features_batch.append(torch.stack(Fs_i))

        # Part II: infer *one* style_style_feature from K of them
        # and infer *one* style_content_residual_features from K of them

        ## Implementation 1: take average of the K style & content features from the style images
        ### Find the average style feature
        # style_style_feature = torch.mean(torch.stack(style_style_feature_list), dim=0)
        ### Find the average content residual features
        # style_content_residual_features[i][j]: i = index of the style image, j = index of residual feature (fs) of its content encoding
        # average_features = []
        # for i in range(len(style_content_residual_features_list[0])):
        #     fsi = [fs[i] for fs in style_content_residual_features_list]
        #     average_features.append(torch.mean(torch.stack(fsi), dim=0))
        # style_content_residual_features = average_features

        ## Implementation 2: take learned feature of the K style & content features from the style images
        style_style_feature, style_content_residual_features = self.config.k_feature_extractor(
            style_features=style_style_feature_batch,
            content_features=style_content_residual_features_batch
        )

        # Part III: Do the rest and run the UNet

        batch_size, channel, height, width = style_style_feature.shape
        style_hidden_states = style_style_feature.permute(0, 2, 3, 1).reshape(batch_size, height*width, channel)

        input_hidden_states = [style_style_feature, content_content_residual_features, \
                               style_hidden_states, style_content_residual_features]

        out = self.config.unet(
            x_t, 
            timesteps, 
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        noise_pred = out[0]
        offset_out_sum = out[1]

        return noise_pred, offset_out_sum

class FontDiffuserModelDPM(ModelMixin, ConfigMixin):
    """DPM Forward function for FontDiffuer with content encoder \
        style encoder and unet.
    """
    @register_to_config
    def __init__(
        self, 
        unet, 
        style_encoder,
        content_encoder,
        k_feature_extractor,
    ):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder
        self.k_feature_extractor = k_feature_extractor

    
    def forward(
        self, 
        x_t, 
        timesteps, 
        cond,
        content_encoder_downsample_size,
        version,
    ):
        content_images = cond[0]
        style_images = cond[1]

        # Part I: Get style and content features from style and content images

        ## Original implementation: one style image
        ### Get style feature from style image
        # style_style_feature, _, style_style_residual_features = self.style_encoder(style_images)

        ### Get content feature from content image
        # content_content_feature, content_content_residual_features = self.content_encoder(content_images)
        # content_content_residual_features.append(content_content_feature)

        ### Get content feature from style image
        # style_content_feature, style_content_residual_features = self.content_encoder(style_images)
        # style_content_residual_features.append(style_content_feature)

        ## Implementation 1: take average of the K style & content features from the style images

        ### Initialization
        # style_image_list = style_images
        # K = len(style_images) // 2
        # uncond_style_list = style_images[0 : K]
        # cond_style_list = style_images[K :]

        ### Get style feature from style image *list*
        # style_style_feature_list=[]
        # for uncond_style, cond_style in zip(uncond_style_list, cond_style_list):
        #     style_style_feature, _, style_style_residual_features = self.style_encoder(torch.stack([uncond_style, cond_style]))
        #     style_style_feature_list.append(style_style_feature)

        ### Get content feature from content image
        # content_content_feture, content_content_residual_features = self.content_encoder(content_images)
        # content_content_residual_features.append(content_content_feture)

        ### Get content feature from style image *list*
        # style_content_residual_features_list=[]
        # for uncond_style, cond_style in zip(uncond_style_list, cond_style_list):
        #     style_content_feature, style_content_residual_features = self.content_encoder(torch.stack([uncond_style, cond_style]))
        #     style_content_residual_features.append(style_content_feature)
        #     style_content_residual_features_list.append(style_content_residual_features)

        ## Implementation 2: take learned feature of the K style & content features from the style images
        
        ### Initialization
        K = len(style_images) // 2
        uncond_style_batch = style_images[0 : K]
        cond_style_batch = style_images[K :]

        ### Get style feature from style image *list*
        uncond_style_style_feature, _, _ = self.config.style_encoder(uncond_style_batch)
        cond_style_style_feature, _, _ = self.config.style_encoder(cond_style_batch)

        ### Get content feature from content image
        content_content_feture, content_content_residual_features = self.config.content_encoder(content_images)
        content_content_residual_features.append(content_content_feture)

        ### Get content feature from style image *list*
        uncond_style_content_feature, uncond_style_content_residual_features = self.config.content_encoder(uncond_style_batch)
        uncond_style_content_residual_features.append(uncond_style_content_feature)
        cond_style_content_feature, cond_style_content_residual_features = self.config.content_encoder(cond_style_batch)
        cond_style_content_residual_features.append(cond_style_content_feature)

        # Part II: infer *one* style_style_feature from K of them
        # and infer *one* style_content_residual_features from K of them

        ## Implementation 1: take average of the K style & content features from the style images
        ### Find the average style feature
        # style_style_feature = torch.mean(torch.stack(style_style_feature_list), dim=0)
        ### Find the average content residual features
        # style_content_residual_features[i][j]: i = index of the style image, j = index of residual feature (fs) of its content encoding
        # average_features = []
        # for i in range(len(style_content_residual_features_list[0])):
        #     fsi = [fs[i] for fs in style_content_residual_features_list]
        #     average_features.append(torch.mean(torch.stack(fsi), dim=0))
        # style_content_residual_features = average_features

        ## Implementation 2: take learned feature of the K style & content features from the style images
        combined_style_style_feature = torch.stack([uncond_style_style_feature, cond_style_style_feature])
        combined_style_content_residual_features = [
            torch.stack([uncond_style_content_residual_features[i], cond_style_content_residual_features[i]])
            for i in range(len(uncond_style_content_residual_features))
        ]
        style_style_feature, style_content_residual_features = self.config.k_feature_extractor(
            style_features=combined_style_style_feature,
            content_features=combined_style_content_residual_features
        )

        # Part III: Do the rest and run the UNet

        batch_size, channel, height, width = style_style_feature.shape
        style_hidden_states = style_style_feature.permute(0, 2, 3, 1).reshape(batch_size, height*width, channel)

        input_hidden_states = [style_style_feature, content_content_residual_features, style_hidden_states, style_content_residual_features]

        out = self.config.unet(
            x_t, 
            timesteps, 
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        noise_pred = out[0]

        return noise_pred
