import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp
from timm.models.layers import trunc_normal_
import math
from models.test_advanced_network.denoiser import  MLPSepConv, SinusoidalEmbedding,MLP
from einops import rearrange
import xformers
from copy import deepcopy
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class MHAttention(nn.Module):
    def __init__(self, is_causal=False, dropout_level=0, n_heads=4):
        super().__init__()
        self.is_causal = is_causal
        self.dropout_level = dropout_level
        self.n_heads = n_heads

    def forward(self, q, k, v, attn_mask=None):

        assert q.size(-1) == k.size(-1)
        assert k.size(-2) == v.size(-2)

        q, k, v = [rearrange(x, 'bs n (d h) -> bs h n d', h=self.n_heads) for x in [q,k,v]]
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        out = nn.functional.scaled_dot_product_attention(q, k, v,
                                                          attn_mask=attn_mask,
                                                          is_causal=self.is_causal,
                                                          dropout_p=self.dropout_level if self.training else 0)

        out = rearrange(out, 'bs h n d -> bs n (d h)', h=self.n_heads)

        return out
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_patches=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rel_pos_bias = RelativePositionBias(
            window_size=[int(num_patches ** 0.5), int(num_patches ** 0.5)], num_heads=num_heads)

    def get_masked_rel_bias(self, B, ids_keep):
        # get masked rel_pos_bias
        rel_pos_bias = self.rel_pos_bias()
        rel_pos_bias = rel_pos_bias.unsqueeze(dim=0).repeat(B, 1, 1, 1)

        rel_pos_bias_masked = torch.gather(
            rel_pos_bias, dim=2, index=ids_keep.unsqueeze(dim=1).unsqueeze(dim=-1).repeat(1, rel_pos_bias.shape[1], 1,
                                                                                          rel_pos_bias.shape[-1]))
        rel_pos_bias_masked = torch.gather(
            rel_pos_bias_masked, dim=3,
            index=ids_keep.unsqueeze(dim=1).unsqueeze(dim=2).repeat(1, rel_pos_bias.shape[1], ids_keep.shape[1], 1))
        return rel_pos_bias_masked
    def zero_padding(self,x):
        N = x.size(-1)
        M = ((N+7)//8)*8
        padding = M-N
        pad_tuple = (0,padding)
        output_tensor = nn.functional.pad(x,pad_tuple,mode='constant',value=0)
        return output_tensor

    def forward(self, x, ids_keep=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        if ids_keep is not None:
            rp_bias = self.get_masked_rel_bias(B, ids_keep)
        else:
            rp_bias = self.rel_pos_bias().unsqueeze(0).repeat(B,1,1,1)
        try:
            x = xformers.ops.memory_efficient_attention(q.permute(0,2,1,3),k.permute(0,2,1,3),v.permute(0,2,1,3),attn_bias=(self.zero_padding(rp_bias).to(q.dtype)[:,:,:,:N])).reshape([B,-1,C])
        except:
            la = 1
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, is_causal=False, dropout_level=0, n_heads=4):
        super().__init__()
        self.kv_linear = nn.Linear(embed_dim, 2*embed_dim, bias=False)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mha = MHAttention(is_causal, dropout_level, n_heads)

    def forward(self, x, y):
        q = self.q_linear(x)
        k, v = self.kv_linear(y).chunk(2, dim=2)
        return self.mha(q,k,v)
class RelativePositionBias(nn.Module):
    # https://github.com/microsoft/unilm/blob/master/beit/modeling_finetune.py
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (
                                             2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - \
                          coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(
                size=(window_size[0] * window_size[1],) * 2, dtype=relative_coords.dtype)
        relative_position_index = relative_coords.sum(-1)

        self.register_buffer("relative_position_index",
                             relative_position_index)

        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self):
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.permute(2, 0, 1).contiguous()


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0,
                                                 end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0]) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1

        labels = torch.where(drop_ids.to(labels.device),
                             self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core MDT Model                                #
#################################################################################
class Control_MDTBlock(nn.Module):
    def __init__(self, block, block_id, embed_dim):
        super().__init__()
        self.block_id = block_id
        self.copied_block = deepcopy(block)
        self.hidden_size = hidden_size = embed_dim
        if block_id == 0:
            self.before_proj = nn.Linear(hidden_size, hidden_size)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)
    def forward(self,x,noise_level,y,c=None,skip=None,ids_keep=None):
        if self.block_id == 0:
            c = self.before_proj(c)
            c = self.copied_block(x+c,noise_level,y,skip,ids_keep)
            c_skip = self.after_proj(c)
        else:
            c = self.copied_block(c,noise_level,y,skip,ids_keep)
            c_skip = self.after_proj(c)
        return c,c_skip
class MDTBlock(nn.Module):
    """
    A MDT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, skip=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        def approx_gelu(): return nn.GELU(approximate="tanh")

        self.mlp = Mlp(in_features=hidden_size,
                       hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.skip_linear = nn.Linear(2 * hidden_size, hidden_size) if skip else None
        self.cross_attn = CrossAttention(hidden_size, n_heads=num_heads)

    def forward(self, x, noise_level,y, skip=None, ids_keep=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            noise_level).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), ids_keep=ids_keep)
        x = x + self.cross_attn(x,y)
        x = x + \
            gate_mlp.unsqueeze(
                1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of MDT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class MDTv2_ControlNet(nn.Module):
    """
    Masked Diffusion Transformer v2.
    """

    def __init__(
            self,
            base_model,
            copy_blocks_num,
            hidden_size=512,
    ):
        super().__init__()
        self.learn_sigma = base_model.learn_sigma
        self.in_channels = base_model.in_channels
        self.out_channels = base_model.out_channels
        self.patch_size = base_model.patch_size
        self.num_heads = base_model.num_heads


        self.x_embedder = base_model.x_embedder

        self.fourier_feats = base_model.fourier_feats
        self.label_proj = base_model.label_proj
        num_patches = base_model.x_embedder.num_patches
        # Will use learnbale sin-cos embedding:
        self.pos_embed = base_model.pos_embed

        #half_depth = (depth - decode_layer) // 2
        self.half_depth = base_model.half_depth

        self.en_inblocks = base_model.en_inblocks
        self.en_outblocks = base_model.en_outblocks
        self.en_inblocks_controlnet = []
        for i in range(len(self.en_inblocks)):
            self.en_inblocks_controlnet.append(Control_MDTBlock(self.en_inblocks[i],i,hidden_size))
        self.en_inblocks_controlnet = nn.ModuleList(self.en_inblocks_controlnet)
        self.en_outblocks_controlnet = []
        for i in range(len(self.en_outblocks)):
            self.en_outblocks_controlnet.append(Control_MDTBlock(self.en_outblocks[i],i+len(self.en_inblocks),hidden_size))
        self.en_outblocks_controlnet = nn.ModuleList(self.en_outblocks_controlnet)
        self.de_blocks = base_model.de_blocks
        self.sideblocks = base_model.sideblocks
        self.final_layer = base_model.final_layer

        self.decoder_pos_embed = base_model.decoder_pos_embed
        self.mask_token = base_model.mask_token
        self.mask_ratio = base_model.mask_ratio
        self.decode_layer = base_model.decode_layer
        print("mask ratio:", self.mask_ratio, "decode_layer:", self.decode_layer)
        self.norm_c = base_model.norm_c
        self.norm_y = base_model.norm_y

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio,c=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        if c is not None:
            c_masked = torch.gather(
                c, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        if c is not None:
            return x_masked, mask, ids_restore, ids_keep,c_masked
        else:
            return x_masked, mask, ids_restore, ids_keep
    def fix_masking(self, x, mask_token,c=None):
        N, L, D = x.shape
        B,H,W = mask_token.shape
        len_keep = int(mask_token.sum())
        mask_token = mask_token.view(B,H*W)
        #note that, here small is keep, large is remove
        mask_token = 1 - mask_token
        ids_shuffle = torch.argsort(mask_token, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        if c is not None:
            c_masked = torch.gather(
                c, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        # mask = torch.ones([N, L], device=x.device)
        # mask[:, :len_keep] = 0
        # # unshuffle to get the binary mask
        # mask = torch.gather(mask, dim=1, index=ids_restore)
        if c is not None:
            return x_masked, mask_token, ids_restore, ids_keep,c_masked
        else:
            return x_masked, mask_token, ids_restore, ids_keep,c
    def forward_side_interpolater(self, x, noise_level,y, mask, ids_restore):
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # add pos embed
        x = x + self.decoder_pos_embed

        # pass to the basic block
        x_before = x
        for sideblock in self.sideblocks:
            x = sideblock(x, noise_level,y, ids_keep=None)

        # masked shortcut
        mask = mask.unsqueeze(dim=-1)
        x = x * mask + (1 - mask) * x_before

        return x
    def forward_c(self,c):
        if c is not None:
            c = self.x_embedder(c) + self.pos_embed
        else:
            c = c
        return c

    def forward_mask(self, x, noise_level, y, enable_mask=False,mask_ratio=None):
        """
        Forward pass of MDT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        enable_mask: Use mask latent modeling
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2

        # t = self.t_embedder(t)[:,0,:]  # (N, D)
        # y = self.y_embedder(y)  # (N, D)
        # c = t + y  # (N, D)
        noise_level = self.fourier_feats(noise_level)
        noise_level = self.norm_c(noise_level)

        y = self.label_proj(y).unsqueeze(1)
        y = self.norm_y(y)
        # c = torch.cat([noise_level, y], dim=1)
        # c = self.norm_c(c)

        input_skip = x

        masked_stage = False
        skips = []
        # masking op for training
        if self.mask_ratio is not None and enable_mask:
            # masking: length -> length * mask_ratio
            rand_mask_ratio = torch.rand(1, device=x.device)  # noise in [0, 1]
            rand_mask_ratio = rand_mask_ratio * 0.2 + self.mask_ratio  # mask_ratio, mask_ratio + 0.2
            # print(rand_mask_ratio)
            x, mask, ids_restore, ids_keep = self.random_masking(
                x, rand_mask_ratio)
            masked_stage = True

        for block in self.en_inblocks:
            if masked_stage:
                x = block(x, noise_level, y, ids_keep=ids_keep)
            else:
                x = block(x, noise_level, y, ids_keep=None)
            skips.append(x)

        for block in self.en_outblocks:
            if masked_stage:
                x = block(x, noise_level, y, skip=skips.pop(), ids_keep=ids_keep)
            else:
                x = block(x, noise_level, y, skip=skips.pop(), ids_keep=None)

        if self.mask_ratio is not None and enable_mask:
            x = self.forward_side_interpolater(x, noise_level, y, mask, ids_restore)
            masked_stage = False
        else:
            # add pos embed
            x = x + self.decoder_pos_embed

        for i in range(len(self.de_blocks)):
            block = self.de_blocks[i]
            this_skip = input_skip

            x = block(x, noise_level, y, skip=this_skip, ids_keep=None)

        x = self.final_layer(x, noise_level)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x
    def forward(self, x, noise_level, y, c=None,mask_token=None,enable_mask=False,not_control=False):
        """
        Forward pass of MDT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        enable_mask: Use mask latent modeling
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        c = self.forward_c(c)
        # t = self.t_embedder(t)[:,0,:]  # (N, D)
        # y = self.y_embedder(y)  # (N, D)
        # c = t + y  # (N, D)
        noise_level = self.fourier_feats(noise_level)
        noise_level = self.norm_c(noise_level)

        y = self.label_proj(y).unsqueeze(1)
        y = self.norm_y(y)
        #c = torch.cat([noise_level, y], dim=1)
        #c = self.norm_c(c)

        input_skip = x

        masked_stage = False
        # masking op for training
        if self.mask_ratio is not None and enable_mask:
            # masking: length -> length * mask_ratio
            rand_mask_ratio = torch.rand(1, device=x.device)  # noise in [0, 1]
            rand_mask_ratio = rand_mask_ratio * 0.2 + self.mask_ratio  # mask_ratio, mask_ratio + 0.2
            # print(rand_mask_ratio)
            x, mask, ids_restore, ids_keep,c = self.random_masking(
                x, rand_mask_ratio,c)
            masked_stage = True
        if mask_token is not None:
            x, mask, ids_restore, ids_keep,c = self.fix_masking(
                x, mask_token,c)
            masked_stage = True
        x_skips = [];
        c_skips = []
        x = self.en_inblocks[0](x, noise_level,y, ids_keep= ids_keep if masked_stage else None)
        x_skips.append(x)
        if isinstance(not_control,list):
            control_mask = torch.tensor(not_control).reshape([-1,1,1]).repeat(1,x.shape[1],x.shape[2]).to(x.device)

        else:
            control_mask = torch.ones_like(x)
        if c is not None:
            for i in range(1,len(self.en_inblocks)):
                c,c_skip = self.en_inblocks_controlnet[i-1](x,noise_level,y,c,ids_keep=ids_keep if masked_stage else None)
                x = x+c_skip*control_mask
                x = self.en_inblocks[i](x,noise_level,y,ids_keep=ids_keep if masked_stage else None)
                x_skips.append(x)
                c_skips.append(c)
            c,c_skip = self.en_inblocks_controlnet[-1](x,noise_level,y,c,ids_keep=ids_keep if masked_stage else None)
            c_skips.append(c)
        else:
            for i in range(1,len(self.en_inblocks)):
                x = self.en_inblocks[i](x,noise_level,y,ids_keep=ids_keep if masked_stage else None)
                x_skips.append(x)
        if c is not None:
            for i in range(0,len(self.en_outblocks)):
                x = x + c_skip*control_mask
                x = self.en_outblocks[i](x, noise_level, y, skip=x_skips.pop(), ids_keep=ids_keep if masked_stage else None)
                c,c_skip = self.en_outblocks_controlnet[i](x, noise_level, y, c,skip=c_skips.pop(), ids_keep=ids_keep if masked_stage else None)
            x = x + c_skip*control_mask
        else:
            for i in range(0,len(self.en_outblocks)):
                x = self.en_outblocks[i](x, noise_level, y, skip=x_skips.pop(), ids_keep=ids_keep if masked_stage else None)

        if masked_stage:
            x = self.forward_side_interpolater(x, noise_level,y, mask, ids_restore)
            masked_stage = False
        else:
            # add pos embed
            x = x + self.decoder_pos_embed

        for i in range(len(self.de_blocks)):
            block = self.de_blocks[i]
            this_skip = input_skip

            x = block(x, noise_level,y, skip=this_skip, ids_keep=None)

        x = self.final_layer(x, noise_level)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale=None, diffusion_steps=1000, scale_pow=4.0):
        """
        Forward pass of MDT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        if cfg_scale is not None:
            half = x[: len(x) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = self.forward(combined, t, y)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

            scale_step = (
                                 1 - torch.cos(
                             ((1 - t / diffusion_steps) ** scale_pow) * math.pi)) * 1 / 2  # power-cos scaling
            real_cfg_scale = (cfg_scale - 1) * scale_step + 1
            real_cfg_scale = real_cfg_scale[: len(x) // 2].view(-1, 1, 1, 1)

            half_eps = uncond_eps + real_cfg_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)
        else:
            model_out = self.forward(x, t, y)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   MDTv2 Configs                               #
#################################################################################

def MDTv2_XL_2(**kwargs):
    return MDTv2(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def MDTv2_L_2(**kwargs):
    return MDTv2(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def MDTv2_B_2(**kwargs):
    return MDTv2(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def MDTv2_S_2(**kwargs):
    return MDTv2(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)
