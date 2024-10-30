import math
import torch
import torch.nn as nn

from diffusers import ModelMixin
from diffusers.configuration_utils import (ConfigMixin, 
                                           register_to_config)

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
    ):
        super().__init__()                      #initialization by ModelMixin
        self.unet = unet                        #unet is the model
        self.style_encoder = style_encoder      #style_encoder is the model
        self.content_encoder = content_encoder
    
    # def takeavg(self, style_img_feature,batch_size,channel,height,width,style_hidden_states):
    #         #take average of the style image feature
    #         avg_style_img_feature = style_img_feature.mean(dim=(2, 3))
    #         avg_style_hidden_states = avg_style_img_feature.unsqueeze(1).repeat(1, height*width, 1)
    #     return 

    def forward(
        self, 
        x_t, 
        timesteps, 
        style_images,
        content_images,
        content_encoder_downsample_size,
    ):
        
        # #get the 5 style images
        style_img_feature = []
        batch_size = []
        channel = []
        height = []
        width = []
        style_hidden_states = []
        for i in range(5):
            style_img_feature[i], _, _ = self.style_encoder(style_images[i])
            #obtain the batch size, channel, height and width of the style image feature
            batch_size[i], channel[i], height[i], width[i] = style_img_feature[i].shape
            #permute the style image feature, the new shape of the tensor is (batch_size, height, width, channel)
            style_hidden_states[i] = style_img_feature[i].permute(0, 2, 3, 1).reshape(batch_size[i], height[i]*width[i], channel[i])



        # # Get the style feature
        # style_img_feature, _, _ = self.style_encoder(style_images)
        # #obtain the batch size, channel, height and width of the style image feature
        # batch_size, channel, height, width = style_img_feature.shape
        # #permute the style image feature, the new shape of the tensor is (batch_size, height, width, channel)
        # style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(batch_size, height*width, channel)
    
        # Get the content feature
        content_img_feature, content_residual_features = self.content_encoder(content_images)
        content_residual_features.append(content_img_feature)
        # Get the content feature from reference image
        style_content_feature, style_content_res_features = self.content_encoder(style_images)
        style_content_res_features.append(style_content_feature)

        input_hidden_states = [style_img_feature, content_residual_features, \
                               style_hidden_states, style_content_res_features]

        out = self.unet(
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
        style_encoder,          #style_encoder from load_fontdiffuser_pipeline
        content_encoder,        #content_encoder from load_fontdiffuser_pipeline
    ):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder
    
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
        style_images_feature=[]
        style_content_res_features=[]

        #style_images[0] is uncond style
        #style_images[1] is cond style
        #for i from 0 to n, style_images[0][i] and style[1][i] change to style_img_feature[0][i] and style_img_feature[1][i]
        for uncond_style, cond_style in zip(style_images[0], style_images[1]):
            feature, _, stlye_res_fea = self.style_encoder(torch.stack([uncond_style, cond_style]))
            style_images_feature.append(feature)
        
        style_images_feature = torch.mean(torch.stack(style_images_feature), dim=0)

        # style_img_feature, _, style_residual_features = (self.style_encoder(style_image) for style_image in style_images)
        
        batch_size, channel, height, width = style_images_feature.shape
        style_hidden_states = style_images_feature.permute(0, 2, 3, 1).reshape(batch_size, height*width, channel)
        
        # Get content feature
        content_img_feture, content_residual_features = self.content_encoder(content_images)
        content_residual_features.append(content_img_feture)
        # Get the content feature from reference image
        for uncond_style, cond_style in zip(style_images[0], style_images[1]):
            con_feature, con_res_feature = self.content_encoder(torch.stack([uncond_style, cond_style]))
            con_res_feature.append(con_feature)
            style_content_res_features.append(con_res_feature)
        
        #style_content_res_features[i][j], i is the index of different style images, j is the index of different fs with the same style image
        #find the average of different style images with the same fs (different i, same j)
        average_features = []
        for i in range(len(style_content_res_features[0])):
            fsi = [fs[i] for fs in style_content_res_features]
            average_features.append(torch.mean(torch.stack(fsi), dim=0))
        
        # for features_at_j in (style_content_res_features):
        #     average_features.append(torch.mean(torch.stack(features_at_j), dim=0))


        # style_content_res_features = torch.mean(torch.stack([torch.stack(features, dim=0) for features in style_content_res_features]), dim=0)

        style_content_res_features = average_features
        # style_content_feature, style_content_res_features = self.content_encoder(style_images)
        # style_content_res_features.append(style_content_feature)

        input_hidden_states = [style_images_feature, content_residual_features, style_hidden_states, style_content_res_features]

        out = self.unet(
            x_t, 
            timesteps, 
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        noise_pred = out[0]
        
        return noise_pred
