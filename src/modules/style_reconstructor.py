import torch
import torch.nn as nn
from .attention_for_reconstructor import SpatialTransformer, ChannelAttnBlock

class StyleReconstructor(nn.Module):
    def __init__(
            self, 
            k_shot,
            max_k=256
        ):
        super().__init__()

        self.k_shot = k_shot
        self.max_k = max_k

        d_embed = 128
        n_heads = 8
        d_head = d_embed // n_heads

        # Style encoding: (B, K=MaxK, C=1024, H=3, W=3) -> (B, K * H * W, C) ; d_query = C
        in_channels = max_k * 3 * 3 # K * H * W
        query_dim = 1024 # C
        # Content encoding (last layer): (B, K=MaxK, C=256, H=12, W=12) -> (B, K * W, C * H)
        context_dim = 256 * 12 # C * H

        # st1_context_channels = max_k * 12 # K * W
        # st2_context_channels = 12 # K * W (from content content residual features, K=1)
        ca1_in_channels = max_k
        ca1_out_channels = 1

        # Spacial transformer 1
        self.st1 = SpatialTransformer(
            in_channels=in_channels,
            n_heads=n_heads,
            d_head=d_head,
            query_dim=query_dim,
            context_dim=context_dim,
        )

        # Spacial transformer 2
        self.st2 = SpatialTransformer(
            in_channels=in_channels,
            n_heads=n_heads,
            d_head=d_head,
            query_dim=query_dim,
            context_dim=context_dim,
        )

        # Channel attention 1
        self.ca1 = ChannelAttnBlock(
            in_channels=ca1_in_channels,
            out_channels=ca1_out_channels,
            non_linearity='silu',
        )

    def forward(self, style_style_feature, style_content_residual_features, content_content_residual_features):
        # for style images style & content feature, if K < maxK, pad with zeros
        # if K > maxK, truncate to maxK

        if style_style_feature.shape[1] < self.max_k:
            style_shape = (style_style_feature.shape[0], self.max_k - style_style_feature.shape[1], *style_style_feature.shape[2:])
            style_style_feature = torch.cat(
                [style_style_feature, torch.zeros(style_shape).to(style_style_feature.device)],
                dim=1,
            )
        elif style_style_feature.shape[1] > self.max_k:
            style_style_feature = style_style_feature[:, :self.max_k, ...]

        for i, scrf in enumerate(style_content_residual_features):
            BB, KK, CC, HH, WW = scrf.shape
            if KK < self.max_k:
                scrf_shape = (BB, self.max_k - KK, CC, HH, WW)
                style_content_residual_features[i] = torch.cat(
                    [scrf, torch.zeros(scrf_shape).to(scrf.device)],
                    dim=1,
                )
            elif KK > self.max_k:
                style_content_residual_features[i] = scrf[:, :self.max_k, ...]

        # print("style_style_feature", style_style_feature.shape)
        # print("style_content_residual_features[-1]", style_content_residual_features[-1].shape)
        # print("content_content_residual_features[-1]", content_content_residual_features[-1].shape)

        B, K, C, H, W = style_style_feature.shape
        ssf = style_style_feature.permute(0, 1, 3, 4, 2).reshape(B, K * H * W, C)

        scrf_final = style_content_residual_features[-1]
        BB, KK, CC, HH, WW = scrf_final.shape
        scrf_final = scrf_final.permute(0, 1, 4, 2, 3).reshape(BB, KK * WW, CC * HH)

        ccrf_final = content_content_residual_features[-1]
        BBB, CCC, HHH, WWW = ccrf_final.shape
        ccrf_final = ccrf_final.permute(0, 3, 1, 2).reshape(BBB, WWW, CCC * HHH)

        valid_ratio = (self.k_shot, self.max_k)

        # print("ssf", ssf.shape)
        # print("scrf_final", scrf_final.shape)
        # print("ccrf_final", ccrf_final.shape)

        st1_output = self.st1(
            hidden_states=ssf,
            context=scrf_final,
            valid_ratio=valid_ratio,
        )
        # print("st1_output", st1_output.shape)
        st2_output = self.st2(
            hidden_states=st1_output,
            context=ccrf_final,
            valid_ratio=valid_ratio,
        )
        # print("st2_output", st2_output.shape)
        unpacked = st2_output.reshape(B, K, H * W, C) # shape it into 3D for channel attention (involving conv2d)
        ca1_output = self.ca1(
            inputs=unpacked,
        )
        # print("ca1_output", ca1_output.shape)

        final_style = ca1_output.reshape(B, H, W, C).permute(0, 3, 1, 2) # back to style feature shape

        # print("final_style", final_style.shape)
        return final_style

if __name__ == "__main__":
    # print number of parameters

    model = StyleReconstructor(k_shot=256, max_k=256)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.numel())

    print("Total:", sum(p.numel() for p in model.parameters() if p.requires_grad))
