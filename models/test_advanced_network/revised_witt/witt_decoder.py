from models.test_advanced_network.revised_witt.witt_modules import *
import torch
from models.test_advanced_network.revised_witt.witt_encoder import SwinTransformerBlock, SwinTransformerBlock_ADALN, AdaptiveModulator
from einops import rearrange
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
        # self.adaLN_modulation = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        # )

    def forward(self, x, c=None):
        #shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        #x = modulate(self.norm_final(x), shift, scale)
        x = self.norm_final(x)
        x = self.linear(x)
        return x
class BasicLayer(nn.Module):

    def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, upsample=None,):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = upsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for _, blk in enumerate(self.blocks):
            x = blk(x)

        if self.upsample is not None:
            x = self.upsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
            print("blk.flops()", blk.flops())
        if self.upsample is not None:
            flops += self.upsample.flops()
            print("upsample.flops()", self.upsample.flops())
        return flops

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        for _, blk in enumerate(self.blocks):
            blk.input_resolution = (H, W)
            blk.update_mask()
        if self.upsample is not None:
            self.upsample.input_resolution = (H, W)


class WITT_Decoder(nn.Module):
    def __init__(self, img_size, embed_dims, depths, num_heads, C,
                 window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,patch_size=2,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,model_type='vit_witt',
                 bottleneck_dim=16):
        super().__init__()

        self.num_layers = len(depths)
        self.ape = ape
        self.embed_dims = embed_dims
        self.patch_norm = patch_norm
        self.num_features = bottleneck_dim
        self.mlp_ratio = mlp_ratio
        self.model_type = model_type
        self.patch_size = patch_size
        self.H = img_size[0]
        self.W = img_size[1]
        self.patches_resolution = (img_size[0]//(patch_size//2) // 2 ** len(depths), img_size[1]//(patch_size//2) // 2 ** len(depths))
        num_patches = self.H // (patch_size)**2 * self.W // (patch_size)**2
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[0]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dims[i_layer]),
                               out_dim=int(embed_dims[i_layer + 1]) if (i_layer < self.num_layers - 1) else 1,
                               input_resolution=(self.patches_resolution[0] * (2 ** i_layer),
                                                 self.patches_resolution[1] * (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               norm_layer=norm_layer,
                               upsample=PatchReverseMerging if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layer)
            print("Decoder ", layer.extra_repr())
        self.head_list = nn.Linear(C, embed_dims[0])
        self.apply(self._init_weights)
        self.hidden_dim = int(self.embed_dims[0] * 1.5)
        self.layer_num = layer_num = 7
        self.bm_list = nn.ModuleList()
        self.sm_list = nn.ModuleList()
        self.sm_list.append(nn.Linear(self.embed_dims[0], self.hidden_dim))
        for i in range(layer_num):
            if i == layer_num - 1:
                outdim = self.embed_dims[0]
            else:
                outdim = self.hidden_dim
            self.bm_list.append(AdaptiveModulator(self.hidden_dim))
            self.sm_list.append(nn.Linear(self.hidden_dim, outdim))
        self.sigmoid = nn.Sigmoid()
        self.final_layer = FinalLayer(hidden_size=self.embed_dims[-1], patch_size=self.patch_size, out_channels=1)
    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = 1
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    def forward(self, x, snr, model='No_WITT'):
        B, L, C = x.size()
        device = x.get_device()
        x = self.head_list(x)

        if self.model_type == 'vit_witt':
            # token modulation according to input snr value
            # snr_cuda = torch.tensor(snr, dtype=torch.float).to(device)
            # snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
            for i in range(self.layer_num):
                if i == 0:
                    temp = self.sm_list[i](x.detach())
                else:
                    temp = self.sm_list[i](temp)
                bm = self.bm_list[i](snr).unsqueeze(1).expand(-1, L, -1)
                temp = temp * bm
            mod_val = self.sigmoid(self.sm_list[-1](temp))
            x = x * mod_val

        for i_layer, layer in enumerate(self.layers):
            x = layer(x)
        x = self.final_layer(x)
        x = self.unpatchify(x)
        #B, L, N = x.shape
        #x = x.reshape(B, self.H, self.W, N).permute(0, 3, 1, 2)
        x = F.sigmoid(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def flops(self):
        flops = 0
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        return flops

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        self.H = H * 2 ** len(self.layers)
        self.W = W * 2 ** len(self.layers)
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(H * (2 ** i_layer),
                                    W * (2 ** i_layer))

class FinalLayer_ADALN(nn.Module):
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

    def forward(self, x, c=None):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.norm_final(x)
        x = self.linear(x)
        return x
class BasicLayer_ADALN(nn.Module):

    def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,cross_attn=False,
                 norm_layer=nn.LayerNorm, upsample=None,):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock_ADALN(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,cross_attn=cross_attn,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = upsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.upsample = None
        self.cross_attn = cross_attn
    def forward(self, x,cond,cross_cond=None):
        for _, blk in enumerate(self.blocks):
            x = blk(x,cond) if self.cross_attn else blk(x,cond,cross_cond)

        if self.upsample is not None:
            x = self.upsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
            print("blk.flops()", blk.flops())
        if self.upsample is not None:
            flops += self.upsample.flops()
            print("upsample.flops()", self.upsample.flops())
        return flops

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        for _, blk in enumerate(self.blocks):
            blk.input_resolution = (H, W)
            blk.update_mask()
        if self.upsample is not None:
            self.upsample.input_resolution = (H, W)

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
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, is_causal=False, dropout_level=0, n_heads=4):
        super().__init__()
        self.kv_linear = nn.Linear(16, 2*embed_dim, bias=False)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mha = MHAttention(is_causal, dropout_level, n_heads)

    def forward(self, x, y):
        q = self.q_linear(x)
        k, v = self.kv_linear(y).chunk(2, dim=2)
        return self.mha(q,k,v)
class WITT_ADALN_Decoder(nn.Module):
    def __init__(self, img_size, embed_dims, depths, num_heads, C,
                 window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,patch_size=2,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,model_type='vit_witt',
                 bottleneck_dim=16):
        super().__init__()

        self.num_layers = len(depths)
        self.ape = ape
        self.embed_dims = embed_dims
        self.patch_norm = patch_norm
        self.num_features = bottleneck_dim
        self.mlp_ratio = mlp_ratio
        self.model_type = model_type
        self.patch_size = patch_size
        self.H = img_size[0]
        self.W = img_size[1]
        self.patches_resolution = (img_size[0]//(patch_size//2) // 2 ** len(depths), img_size[1]//(patch_size//2) // 2 ** len(depths))
        num_patches = self.H // (patch_size)**2 * self.W // (patch_size)**2
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[0]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer_ADALN(dim=int(embed_dims[i_layer]),
                               out_dim=int(embed_dims[i_layer + 1]) if (i_layer < self.num_layers - 1) else 1,
                               input_resolution=(self.patches_resolution[0] * (2 ** i_layer),
                                                 self.patches_resolution[1] * (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               norm_layer=norm_layer,
                               upsample=PatchReverseMerging if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layer)
            print("Decoder ", layer.extra_repr())
        self.head_list = Head_layer( C,embed_dims[0])
        self.head_snr_embedding = SNREmbedder(C); self.head_cr_embedding = SNREmbedder(C);
        self.hidden_dim = int(self.embed_dims[0] * 1.5)
        self.layer_num = layer_num = 7
        self.snr_embedding = nn.ModuleList()
        self.cr_embedding = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.snr_embedding.append(SNREmbedder(embed_dims[i_layer]))
        for i_layer in range(self.num_layers):
            self.cr_embedding.append(SNREmbedder(embed_dims[i_layer]))
        self.sigmoid = nn.Sigmoid()
        self.final_layer = FinalLayer_ADALN(hidden_size=self.embed_dims[-1], patch_size=self.patch_size, out_channels=1)
        self.apply(self._init_weights)
        #self.cross_module = CrossAttention(256, n_heads=4)
        self.ct_module = nn.ModuleList()
        for i_layer in range(2):
            self.ct_module.append(nn.LayerNorm(embed_dims[i_layer], elementwise_affine=False, eps=1e-6))
            self.ct_module.append(CrossAttention(embed_dims[i_layer], n_heads=4))
        self.init_adaln_weights_embedding()
    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = 1
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    def forward(self, x, snr, cr,snr_embeding_list,image_noisy_feature=None):
        B, L, C = x.size()
        device = x.get_device()
        x = self.head_list(x,self.head_snr_embedding(snr[:,0])+self.head_cr_embedding(cr[:,0]))
        # if image_noisy_feature is not None:
        #     x = x + self.cross_module(x,image_noisy_feature.reshape([B,16,256]).permute(0,2,1))
        #snr_embeding_list.reverse()
        if image_noisy_feature is not None:
            image_noisy_feature = image_noisy_feature.reshape([B,16,256]).permute(0,2,1)
        snr_embeding_list_decoder = [];
        for i_layer, layer in enumerate(self.snr_embedding):
            snr_embeding_list_decoder.append(layer(snr[:, 0]) + self.cr_embedding[i_layer](cr[:, 0]))
        for i_layer, layer in enumerate(self.layers):
            if i_layer < 2 and image_noisy_feature is not None:
                x = self.ct_module[i_layer*2](x)
                x = x + self.ct_module[i_layer*2+1](x,image_noisy_feature)
            x = layer(x,snr_embeding_list_decoder[i_layer],cross_cond=image_noisy_feature)

        x = self.final_layer(x,snr_embeding_list_decoder[-1])
        x = self.unpatchify(x)
        #B, L, N = x.shape
        #x = x.reshape(B, self.H, self.W, N).permute(0, 3, 1, 2)
        x = F.sigmoid(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                try:
                    nn.init.constant_(m.bias, 0)
                except:
                    pass
        elif isinstance(m, nn.LayerNorm):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except:
                pass
    def init_adaln_weights_embedding(self):
        nn.init.normal_(self.head_snr_embedding.mlp[0].weight, std=0.02)
        nn.init.normal_(self.head_snr_embedding.mlp[2].weight, std=0.02)
        nn.init.normal_(self.head_cr_embedding.mlp[0].weight, std=0.02)
        nn.init.normal_(self.head_cr_embedding.mlp[2].weight, std=0.02)
        for i_layer, layer in enumerate(self.layers):
            for i_block, block in enumerate(layer.blocks):
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.head_list.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.head_list.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def flops(self):
        flops = 0
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        return flops

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        self.H = H * 2 ** len(self.layers)
        self.W = W * 2 ** len(self.layers)
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(H * (2 ** i_layer),
                                    W * (2 ** i_layer))

def create_decoder(image_dims,C,patch_size,model_type,window_size,embed_dims,depths=[2,2,6,2],num_heads=[4,6,8,10]):
    embed_dims.reverse()
    depths.reverse()
    num_heads.reverse()
    kwargs = dict(
            img_size=(image_dims, image_dims),patch_size=patch_size,
            embed_dims=embed_dims, depths=depths, num_heads=num_heads,
            C=C, window_size=window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,model_type=model_type,
        )
    if 'adaln' in model_type:
        model = WITT_ADALN_Decoder(**kwargs)
    else:
        model = WITT_Decoder(**kwargs)
    return model



def build_model(config):
    input_image = torch.ones([1, 1536, 256]).to(config.device)
    model = create_decoder(**config.encoder_kwargs).to(config.device)
    t0 = datetime.datetime.now()
    with torch.no_grad():
        for i in range(100):
            features = model(input_image, SNR=15)
        t1 = datetime.datetime.now()
        delta_t = t1 - t0
        print("Decoding Time per img {}s".format((delta_t.seconds + 1e-6 * delta_t.microseconds) / 100))
    print("TOTAL FLOPs {}G".format(model.flops() / 10 ** 9))
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print("TOTAL Params {}M".format(num_params / 10 ** 6))
