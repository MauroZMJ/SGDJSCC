from models.test_advanced_network.revised_witt.witt_modules import *
import torch


class SwinTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,

                 mlp_ratio=4., qkv_bias=True, qk_scale=None, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        if self.shift_size >= self.window_size:
            la = 1
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):

        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        B_, N, C = x_windows.shape

        # merge windows
        attn_windows = self.attn(x_windows,
                                 add_token=False,
                                 mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

    def update_mask(self):
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            self.attn_mask = attn_mask.cuda()
        else:
            pass


class BasicLayer(nn.Module):
    def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm,
                 downsample=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=out_dim,
                                 input_resolution=(input_resolution[0] // 2, input_resolution[1] // 2),
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        for _, blk in enumerate(self.blocks):
            x = blk(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def update_resolution(self, H, W):
        for _, blk in enumerate(self.blocks):
            blk.input_resolution = (H, W)
            blk.update_mask()
        if self.downsample is not None:
            self.downsample.input_resolution = (H * 2, W * 2)


class AdaptiveModulator(nn.Module):
    def __init__(self, M):
        super(AdaptiveModulator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.Sigmoid()
        )

    def forward(self, snr):
        return self.fc(snr)


class WITT_Encoder(nn.Module):
    def __init__(self, img_size, patch_size, in_chans,
                 embed_dims, depths, num_heads, C,
                 window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, patch_norm=True, model_type='WITT',
                 bottleneck_dim=16):
        super().__init__()
        self.num_layers = len(depths)
        self.patch_norm = patch_norm
        self.num_features = bottleneck_dim
        self.mlp_ratio = mlp_ratio
        self.embed_dims = embed_dims
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.patches_resolution = (img_size[0] // (patch_size // 2), img_size[1] // (patch_size // 2))
        self.model_type = model_type
        self.H = img_size[0] // (patch_size // 2) // (2 ** self.num_layers)
        self.W = img_size[1] // (patch_size // 2) // (2 ** self.num_layers)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dims[0])
        self.hidden_dim = int(self.embed_dims[len(embed_dims) - 1] * 1.5)
        self.layer_num = layer_num = 7
        self.bm_list = nn.ModuleList()
        self.sm_list = nn.ModuleList()
        self.sm_list.append(nn.Linear(self.embed_dims[len(embed_dims) - 1], self.hidden_dim))
        for i in range(layer_num):
            if i == layer_num - 1:
                outdim = self.embed_dims[len(embed_dims) - 1]
            else:
                outdim = self.hidden_dim
            self.bm_list.append(AdaptiveModulator(self.hidden_dim))
            self.sm_list.append(nn.Linear(self.hidden_dim, outdim))
        self.sigmoid = nn.Sigmoid()

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dims[i_layer - 1]) if i_layer != 0 else 3,
                               out_dim=int(embed_dims[i_layer]),
                               input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                                 self.patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               norm_layer=norm_layer,
                               downsample=PatchMerging if i_layer != 0 else None)
            print("Encoder ", layer.extra_repr())
            self.layers.append(layer)
        self.norm = norm_layer(embed_dims[-1])
        self.head_list = nn.Linear(embed_dims[-1], C)
        self.apply(self._init_weights)
        self.hidden_dim_ra = int(C * 1.5)
        self.bm_list1 = nn.ModuleList()
        self.sm_list1 = nn.ModuleList()
        self.sm_list1.append(nn.Linear(C, self.hidden_dim_ra))
        for i in range(layer_num):
            if i == layer_num - 1:
                outdim = C
            else:
                outdim = self.hidden_dim_ra
            self.bm_list1.append(AdaptiveModulator(self.hidden_dim_ra))
            self.sm_list1.append(nn.Linear(self.hidden_dim_ra, outdim))
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, x, snr, cr):
        B, C, H, W = x.size()
        device = x.get_device()
        x = self.patch_embed(x)
        for i_layer, layer in enumerate(self.layers):
            x = layer(x)
        x = self.norm(x)

        # if self.model_type == 'vit_witt':
        #     # snr_cuda = torch.tensor(snr, dtype=torch.float).to(device)
        #     # snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)

        for i in range(self.layer_num):
            if i == 0:
                temp = self.sm_list[i](x.detach())
            else:
                temp = self.sm_list[i](temp)
            bm = self.bm_list[i](snr).unsqueeze(1).expand(-1, self.H * self.W, -1)
            temp = temp * bm
        mod_val = self.sigmoid(self.sm_list[-1](temp))
        x = x * mod_val
        x = self.head_list(x)
        if 'ra' in self.model_type:
            for i in range(self.layer_num):
                if i == 0:
                    temp = self.sm_list1[i](x.detach())
                else:
                    temp = self.sm_list1[i](temp)

                bm = self.bm_list1[i](cr).unsqueeze(1).expand(-1, H * W // (self.num_layers ** 4), -1)
                temp = temp * bm
            mod_val = self.sigmoid(self.sm_list1[-1](temp))
            x = x * mod_val

            importance_metric = torch.sum(mod_val, dim=1)
            sorted, indices = importance_metric.sort(dim=1, descending=True)
            indice_sequence = torch.arange(x.shape[2]).expand(B, x.shape[2]).to(x.device)
            threshold = cr.expand_as(indice_sequence)
            mask_ori = indice_sequence < threshold
            mask = torch.zeros_like(importance_metric).reshape(-1)
            add = torch.Tensor(range(0, B * x.size()[2], x.size()[2])).unsqueeze(1).repeat(1, x.size()[2])
            c_indices = indices + add.int().cuda()
            mask[c_indices.reshape(-1)] = mask_ori.reshape(-1).to(x.dtype)
            mask = mask.reshape(B, x.size()[2])
            mask = mask.unsqueeze(1).expand(-1, H * W // (self.num_layers ** 4), -1)
            x = x * mask
        if 'ra' in self.model_type:
            return x, mask
        else:
            return x, None

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
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        return flops

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(H // (2 ** (i_layer + 1)),
                                    W // (2 ** (i_layer + 1)))


class SwinTransformerBlock_ADALN(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,cross_attn=False,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        if self.shift_size >= self.window_size:
            la = 1
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        self.cross_attn = cross_attn
        if self.cross_attn:
            la = 1
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, cond,cross_cond=None):

        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            cond).chunk(6, dim=1)

        shortcut = x
        x = modulate(self.norm1(x), shift_msa, scale_msa)

        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        B_, N, C = x_windows.shape

        # merge windows
        attn_windows = self.attn(x_windows,
                                 add_token=False,
                                 mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + gate_msa.unsqueeze(1) * x
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

    def update_mask(self):
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            self.attn_mask = attn_mask.cuda()
        else:
            pass


class BasicLayer_ADALN(nn.Module):
    def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm,
                 downsample=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.blocks = nn.ModuleList([
            SwinTransformerBlock_ADALN(dim=out_dim,
                                       input_resolution=(input_resolution[0] // 2, input_resolution[1] // 2),
                                       num_heads=num_heads, window_size=window_size,
                                       shift_size=0 if (i % 2 == 0) else window_size // 2,
                                       mlp_ratio=mlp_ratio,
                                       qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, cond):
        if self.downsample is not None:
            x = self.downsample(x)
        for _, blk in enumerate(self.blocks):
            x = blk(x, cond)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def update_resolution(self, H, W):
        for _, blk in enumerate(self.blocks):
            blk.input_resolution = (H, W)
            blk.update_mask()
        if self.downsample is not None:
            self.downsample.input_resolution = (H * 2, W * 2)

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
class WITT_ADALN_Encoder(nn.Module):
    def __init__(self, img_size, patch_size, in_chans,
                 embed_dims, depths, num_heads, C,
                 window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, patch_norm=True, model_type='WITT',uncertain=False,
                 bottleneck_dim=16):
        super().__init__()
        self.num_layers = len(depths)
        self.patch_norm = patch_norm
        self.num_features = bottleneck_dim
        self.mlp_ratio = mlp_ratio
        self.embed_dims = embed_dims
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.patches_resolution = (img_size[0] // (patch_size // 2), img_size[1] // (patch_size // 2))
        self.model_type = model_type
        self.H = img_size[0] // (patch_size // 2) // (2 ** self.num_layers)
        self.W = img_size[1] // (patch_size // 2) // (2 ** self.num_layers)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dims[0])
        self.uncertain = uncertain
        if self.uncertain:
            self.uncertain_patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dims[0])
        else:
            self.uncertain_patch_embed = None
        self.hidden_dim = int(self.embed_dims[len(embed_dims) - 1] * 1.5)
        self.layer_num = layer_num = 7
        self.bm_list = nn.ModuleList()
        self.sm_list = nn.ModuleList()
        self.sm_list.append(nn.Linear(self.embed_dims[len(embed_dims) - 1], self.hidden_dim))
        for i in range(layer_num):
            if i == layer_num - 1:
                outdim = self.embed_dims[len(embed_dims) - 1]
            else:
                outdim = self.hidden_dim
            self.bm_list.append(AdaptiveModulator(self.hidden_dim))
            self.sm_list.append(nn.Linear(self.hidden_dim, outdim))
        self.sigmoid = nn.Sigmoid()

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer_ADALN(dim=int(embed_dims[i_layer - 1]) if i_layer != 0 else 3,
                                     out_dim=int(embed_dims[i_layer]),
                                     input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                                       self.patches_resolution[1] // (2 ** i_layer)),
                                     depth=depths[i_layer],
                                     num_heads=num_heads[i_layer],
                                     window_size=window_size,
                                     mlp_ratio=self.mlp_ratio,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     norm_layer=norm_layer,
                                     downsample=PatchMerging if i_layer != 0 else None)
            print("Encoder ", layer.extra_repr())
            self.layers.append(layer)
        self.norm = norm_layer(embed_dims[-1])
        self.head_list = Head_layer(embed_dims[-1], C)
        self.snr_embedding = nn.ModuleList()
        self.cr_embedding = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.snr_embedding.append(SNREmbedder(embed_dims[i_layer]))
        for i_layer in range(self.num_layers):
            self.cr_embedding.append(SNREmbedder(embed_dims[i_layer]))
        self.apply(self._init_weights)

        self.hidden_dim_ra = int(C * 1.5)
        self.bm_list1 = nn.ModuleList()
        self.sm_list1 = nn.ModuleList()
        self.sm_list1.append(nn.Linear(C, self.hidden_dim_ra))
        for i in range(layer_num):
            if i == layer_num - 1:
                outdim = C
            else:
                outdim = self.hidden_dim_ra
            self.bm_list1.append(AdaptiveModulator(self.hidden_dim_ra))
            self.sm_list1.append(nn.Linear(self.hidden_dim_ra, outdim))
        self.sigmoid1 = nn.Sigmoid()
        self.pos_embed = nn.Parameter(torch.zeros(
            1, 4096, 128), requires_grad=True)
        pos_embed_init = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(4096 ** 0.5))
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed_init).float().unsqueeze(0))
        self.norm_y = nn.LayerNorm(128)


    def forward(self, x, snr, cr):
        B, C, H, W = x.size()
        device = x.get_device()
        if self.uncertain:
            x = self.patch_embed(x[:,0:1,:]) + self.uncertain_patch_embed(x[:,1:2,:])
        else:
            x = self.patch_embed(x)
        #new added
        x = x + self.pos_embed
        x = self.norm_y(x)
        snr_embeding_list = [];
        cr_embedding_list = []
        for i_layer, layer in enumerate(self.snr_embedding):
            snr_embeding_list.append(self.cr_embedding[i_layer](cr[:, 0]))
            #snr_embeding_list.append(layer(snr[:, 0]) + self.cr_embedding[i_layer](cr[:, 0]))

        for i_layer, layer in enumerate(self.layers):
            x = layer(x, snr_embeding_list[i_layer])
        x = self.norm(x)

        x = self.head_list(x, snr_embeding_list[-1])

        if 'ra' in self.model_type:
            for i in range(self.layer_num):
                if i == 0:
                    temp = self.sm_list1[i](x.detach())
                else:
                    temp = self.sm_list1[i](temp)

                bm = self.bm_list1[i](cr).unsqueeze(1).expand(-1, H * W // (
                            2 ** (self.num_layers * 2) * (self.patch_size // 2) ** 2), -1)
                temp = temp * bm
            mod_val = self.sigmoid(self.sm_list1[-1](temp))
            x = x * mod_val
            index_tensor = torch._dim_arange(x,dim=2).reshape([1,x.shape[2]]).repeat(B,1)
            mask = (index_tensor < cr.repeat(1,x.shape[2])).reshape(B,1,x.shape[2]).repeat(1,x.shape[1],1).int()
            # importance_metric = torch.sum(mod_val, dim=1)
            # sorted, indices = importance_metric.sort(dim=1, descending=True)
            # indice_sequence = torch.arange(x.shape[2]).expand(B, x.shape[2]).to(x.device)
            #
            # threshold = cr.expand_as(indice_sequence)
            # mask_ori = indice_sequence < threshold
            # mask = torch.zeros_like(importance_metric).reshape(-1)
            # add = torch.Tensor(range(0, B * x.size()[2], x.size()[2])).unsqueeze(1).repeat(1, x.size()[2])
            # c_indices = indices + add.int().cuda()
            # mask[c_indices.reshape(-1)] = mask_ori.reshape(-1).to(x.dtype)
            # mask = mask.reshape(B, x.size()[2])
            # mask = mask.unsqueeze(1).expand(-1, H * W // ((2 ** (self.num_layers * 2)) * (self.patch_size // 2) ** 2),
            #                                 -1)
            x = x * mask
        if 'ra' in self.model_type:
            return x, snr_embeding_list, mask
        else:
            return x, snr_embeding_list, None

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        self.init_adaln_weights_embedding()

    def init_adaln_weights_embedding(self):
        for i_layer, layer in enumerate(self.snr_embedding):
            nn.init.normal_(layer.mlp[0].weight, std=0.02)
            nn.init.normal_(layer.mlp[2].weight, std=0.02)
        for i_layer, layer in enumerate(self.layers):
            for i_block, block in enumerate(layer.blocks):
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.head_list.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.head_list.adaLN_modulation[-1].bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        return flops

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(H // (2 ** (i_layer + 1)),
                                    W // (2 ** (i_layer + 1)))


def create_encoder(image_dims, C, model_type, patch_size, window_size=8, embed_dims=None, depths=[2, 2, 6, 2],uncertain=False,
                   num_heads=[4, 6, 8, 10]):
    kwargs = dict(
        img_size=(image_dims, image_dims), patch_size=patch_size, in_chans=1,
        embed_dims=embed_dims, depths=depths, num_heads=num_heads,
        C=C, window_size=window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, patch_norm=True, model_type=model_type,uncertain=uncertain,
    )
    if 'adaln' in model_type:
        model = WITT_ADALN_Encoder(**kwargs)
    else:
        model = WITT_Encoder(**kwargs)
    return model


def build_model(config):
    input_image = torch.ones([1, 256, 256]).to(config.device)
    model = create_encoder(**config.encoder_kwargs)
    model(input_image)
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print("TOTAL Params {}M".format(num_params / 10 ** 6))
    print("TOTAL FLOPs {}G".format(model.flops() / 10 ** 9))