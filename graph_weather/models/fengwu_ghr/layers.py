"""Building blocks of the FengWu-GHR model.

Contains the vision transformer layers (patch embedding, attention, feed-forward) used
to process data laid out as a regular image, the k-nearest-neighbour interpolation that
moves features between an irregular set of lat/lon points and that regular grid, and the
LoRA layers used to fine-tune a pretrained model.
"""

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from torch_geometric.nn.pool import knn
from torch_geometric.utils import scatter


def pair(t):
    """Duplicate a value into a pair, leaving tuples untouched.

    Args:
        t: either a tuple, which is returned unchanged, or a single value to repeat.

    Returns:
        The input itself if it already is a tuple, the pair (t, t) otherwise.
    """
    return t if isinstance(t, tuple) else (t, t)


def knn_interpolate(
    x: torch.Tensor, pos_x: torch.Tensor, pos_y: torch.Tensor, k: int = 4, num_workers: int = 1
):
    """Interpolate features from one set of positions onto another one.

    Every target position takes the average of the features of its k nearest source
    positions, weighted by the inverse of the squared distance to them.

    Args:
        x (torch.Tensor): features of the source points, of shape
            (num_source_points, num_features).
        pos_x (torch.Tensor): coordinates of the source points, of shape
            (num_source_points, num_dimensions).
        pos_y (torch.Tensor): coordinates of the target points, of shape
            (num_target_points, num_dimensions).
        k (int): number of neighbours averaged for each target point. Defaults to 4.
        num_workers (int): number of workers used by the neighbour search. Defaults to 1.

    Returns:
        torch.Tensor: interpolated features, of shape (num_target_points, num_features).
    """
    with torch.no_grad():
        assign_index = knn(pos_x, pos_y, k, num_workers=num_workers)
        y_idx, x_idx = assign_index[0], assign_index[1]
        diff = pos_x[x_idx] - pos_y[y_idx]
        squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
        weights = 1.0 / torch.clamp(squared_distance, min=1e-16)

        y_idx, x_idx = y_idx.to(x.device), x_idx.to(x.device)
        weights = weights.to(x.device)

    den = scatter(weights, y_idx, 0, pos_y.size(0), reduce="sum")
    y = scatter(x[x_idx] * weights, y_idx, 0, pos_y.size(0), reduce="sum")

    y = y / den

    return y


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    """Build a two dimensional sine-cosine positional embedding.

    Each of the h * w positions of the grid is encoded with the sines and the cosines of
    its row and column indices at dim // 4 geometrically spaced frequencies, so that dim
    has to be a multiple of four.

    Args:
        h (int): number of rows of the grid.
        w (int): number of columns of the grid.
        dim (int): size of the embedding, must be divisible by 4.
        temperature (int): base of the geometric progression of the frequencies.
            Defaults to 10000.
        dtype (torch.dtype): dtype of the returned embedding. Defaults to torch.float32.

    Returns:
        torch.Tensor: positional embedding of shape (h * w, dim).
    """
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


# classes


class FeedForward(nn.Module):
    """Two layer perceptron with a GELU activation, applied to layer normalised input."""

    def __init__(self, dim, hidden_dim):
        """Initialize FeedForward.

        Args:
            dim (int): size of both the input and the output features.
            hidden_dim (int): size of the hidden layer.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        """Apply the perceptron to the input.

        Args:
            x (torch.Tensor): input tensor of shape (..., dim).

        Returns:
            torch.Tensor: output tensor of shape (..., dim).
        """
        return self.net(x)


class Attention(nn.Module):
    """Multi-head self attention, applied to layer normalised input."""

    def __init__(self, dim, heads=8, dim_head=64):
        """Initialize Attention.

        Args:
            dim (int): size of both the input and the output features.
            heads (int): number of attention heads. Defaults to 8.
            dim_head (int): size of each attention head. Defaults to 64.
        """
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        """Let every token of the sequence attend to all the others.

        Args:
            x (torch.Tensor): input tensor of shape (batch, sequence, dim).

        Returns:
            torch.Tensor: output tensor of shape (batch, sequence, dim).
        """
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    """Stack of attention and feed-forward blocks joined by residual connections.

    When res is set, each block is followed by a further attention taken across the
    windows a single image was split into, so that tokens sitting in different windows
    can still see each other.
    """

    def __init__(
        self, dim, depth, heads, dim_head, mlp_dim, res=False, image_size=None, scale_factor=None
    ):
        """Initialize Transformer.

        Args:
            dim (int): size of the token features.
            depth (int): number of attention and feed-forward blocks.
            heads (int): number of attention heads of each block.
            dim_head (int): size of each attention head.
            mlp_dim (int): size of the hidden layer of the feed-forward blocks.
            res (bool): whether to add the attention across windows. Defaults to False.
            image_size (int or tuple): number of rows and columns of tokens of a single
                window, required when res is True. Defaults to None.
            scale_factor (int or tuple): number of windows along the rows and along the
                columns, required when res is True. Defaults to None.
        """
        super().__init__()
        self.depth = depth
        self.res = res
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        self.res_layers = nn.ModuleList([])
        for _ in range(self.depth):
            self.layers.append(
                nn.ModuleList(
                    [Attention(dim, heads=heads, dim_head=dim_head), FeedForward(dim, mlp_dim)]
                )
            )
            if self.res:
                assert (
                    image_size is not None and scale_factor is not None
                ), "If res=True, you must provide h, w and scale_factor"
                h, w = pair(image_size)
                s_h, s_w = pair(scale_factor)
                self.res_layers.append(
                    nn.ModuleList(
                        [  # reshape to original shape     window partition operation
                            #  (b s_h s_w) (h w) d -> b (s_h h) (s_w w) d  -> (b h w) (s_h s_w) d
                            Rearrange(
                                "(b s_h s_w) (h w) d -> (b h w) (s_h s_w) d",
                                h=h,
                                w=w,
                                s_h=s_h,
                                s_w=s_w,
                            ),
                            # TODO ?????
                            Attention(dim, heads=heads, dim_head=dim_head),
                            # restore shape
                            Rearrange(
                                "(b h w) (s_h s_w) d -> (b s_h s_w) (h w) d",
                                h=h,
                                w=w,
                                s_h=s_h,
                                s_w=s_w,
                            ),
                        ]
                    )
                )

    def forward(self, x):
        """Run the input through every block of the stack.

        Args:
            x (torch.Tensor): input tensor of shape (batch, sequence, dim).

        Returns:
            torch.Tensor: normalised output tensor of shape (batch, sequence, dim).
        """
        for i in range(self.depth):
            attn, ff = self.layers[i]
            x = attn(x) + x
            x = ff(x) + x
            if self.res:
                reshape, loc_attn, restore = self.res_layers[i]
                x = reshape(x)
                x = loc_attn(x) + x
                x = restore(x)
        return self.norm(x)


class ImageMetaModel(nn.Module):
    """Vision transformer operating on data laid out as a regular image.

    The image is cut into patches, each patch is embedded and summed with a sine-cosine
    positional embedding, the resulting tokens are processed by a Transformer and finally
    folded back into an image of the same shape as the input.
    """

    def __init__(
        self,
        *,
        image_size,
        patch_size,
        depth,
        heads,
        mlp_dim,
        channels,
        dim_head,
        res=False,
        scale_factor=None,
        **kwargs,
    ):
        """Initialize ImageMetaModel.

        Args:
            image_size (int or tuple): number of rows and columns of the input image.
            patch_size (int or tuple): number of rows and columns of a single patch, it
                has to divide the image size.
            depth (int): number of blocks of the transformer.
            heads (int): number of attention heads of each block.
            mlp_dim (int): size of the hidden layer of the feed-forward blocks.
            channels (int): number of channels of the input image.
            dim_head (int): size of each attention head.
            res (bool): whether the transformer also attends across windows, which is
                what the wrappers below rely on. Defaults to False.
            scale_factor (int or tuple): number of windows along the rows and along the
                columns, required when res is True. Defaults to None.
            **kwargs: further keyword arguments are ignored, so that a model can be
                rebuilt from the attributes of another one.
        """
        super().__init__()
        # TODO this can probably be done better
        self.image_size = image_size
        self.patch_size = patch_size
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.channels = channels
        self.dim_head = dim_head
        self.res = res
        self.scale_factor = scale_factor

        self.image_height, self.image_width = pair(image_size)
        self.patch_height, self.patch_width = pair(patch_size)
        s_h, s_w = pair(scale_factor)

        if res:
            assert scale_factor is not None, "If res=True, you must provide scale_factor"
        assert (
            self.image_height % self.patch_height == 0 and self.image_width % self.patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        patch_dim = channels * self.patch_height * self.patch_width
        dim = patch_dim
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p_h) (w p_w) -> b (h w) (p_h p_w c)",
                p_h=self.patch_height,
                p_w=self.patch_width,
            ),
            nn.LayerNorm(patch_dim),  # TODO Do we need this?
            nn.Linear(patch_dim, dim),  # TODO Do we need this?
            nn.LayerNorm(dim),  # TODO Do we need this?
        )

        self.pos_embedding = posemb_sincos_2d(
            h=self.image_height // self.patch_height,
            w=self.image_width // self.patch_width,
            dim=dim,
        )

        self.transformer = Transformer(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            res=res,
            image_size=(
                self.image_height // self.patch_height,
                self.image_width // self.patch_width,
            ),
            scale_factor=(s_h, s_w),
        )

        self.reshaper = nn.Sequential(
            Rearrange(
                "b (h w) (p_h p_w c) -> b c (h p_h) (w p_w)",
                h=self.image_height // self.patch_height,
                w=self.image_width // self.patch_width,
                p_h=self.patch_height,
                p_w=self.patch_width,
            )
        )

    def forward(self, x):
        """Embed the image into patches, run the transformer and fold the tokens back.

        Args:
            x (torch.Tensor): input image of shape (batch, channels, height, width).

        Returns:
            torch.Tensor: output image of the same shape as the input.
        """
        assert x.shape[1] == self.channels, "Wrong number of channels"
        device = x.device
        dtype = x.dtype

        x = self.to_patch_embedding(x)
        x += self.pos_embedding.to(device, dtype=dtype)

        x = self.transformer(x)
        x = self.reshaper(x)

        return x


class WrapperImageModel(nn.Module):
    """Reuse a trained ImageMetaModel on an image of a higher resolution.

    The image is split into scale_factor windows that are stacked along the batch
    dimension, so that each of them has the resolution the wrapped model was built for.
    The weights of that model are loaded into a new one whose transformer also attends
    across the windows.
    """

    def __init__(self, image_meta_model: ImageMetaModel, scale_factor):
        """Initialize WrapperImageModel.

        Args:
            image_meta_model (ImageMetaModel): model whose settings and weights are reused.
            scale_factor (int or tuple): number of windows along the rows and along the
                columns.
        """
        super().__init__()
        s_h, s_w = pair(scale_factor)
        self.batcher = Rearrange("b c (h s_h) (w s_w) -> (b s_h s_w) c h w", s_h=s_h, s_w=s_w)

        imm_args = vars(image_meta_model)
        imm_args.update({"res": True, "scale_factor": scale_factor})
        self.image_meta_model = ImageMetaModel(**imm_args)
        self.image_meta_model.load_state_dict(image_meta_model.state_dict(), strict=False)

        self.debatcher = Rearrange("(b s_h s_w) c h w -> b c (h s_h) (w s_w)", s_h=s_h, s_w=s_w)

    def forward(self, x):
        """Split the image into windows, run the wrapped model and stitch them back.

        Args:
            x (torch.Tensor): input image of shape (batch, channels, height, width).

        Returns:
            torch.Tensor: output image of the same shape as the input.
        """
        x = self.batcher(x)
        x = self.image_meta_model(x)
        x = self.debatcher(x)
        return x


class MetaModel(nn.Module):
    """ImageMetaModel applied to data given on an irregular set of lat/lon points.

    The features of the points are interpolated onto a regular latitude/longitude grid,
    processed as an image and interpolated back onto the original points.
    """

    def __init__(
        self,
        lat_lons: list,
        *,
        image_size,
        patch_size,
        depth,
        heads,
        mlp_dim,
        channels,
        dim_head=64,
    ):
        """Initialize MetaModel.

        Args:
            lat_lons (list): latitude and longitude of each point of the input.
            image_size (int or tuple): number of rows and columns of the grid the points
                are interpolated onto.
            patch_size (int or tuple): number of rows and columns of a single patch.
            depth (int): number of blocks of the transformer.
            heads (int): number of attention heads of each block.
            mlp_dim (int): size of the hidden layer of the feed-forward blocks.
            channels (int): number of channels of the input.
            dim_head (int): size of each attention head. Defaults to 64.
        """
        super().__init__()
        self.i_h, self.i_w = pair(image_size)

        self.pos_x = torch.tensor(lat_lons).to(torch.long)
        self.pos_y = torch.cartesian_prod(
            (torch.arange(-self.i_h / 2, self.i_h / 2, 1) / self.i_h * 180).to(torch.long),
            (torch.arange(0, self.i_w, 1) / self.i_w * 360).to(torch.long),
        )

        self.image_meta_model = ImageMetaModel(
            image_size=image_size,
            patch_size=patch_size,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            channels=channels,
            dim_head=dim_head,
        )

    def forward(self, x):
        """Interpolate the points onto the grid, run the model and interpolate back.

        Args:
            x (torch.Tensor): input features of shape (batch, num_points, channels).

        Returns:
            torch.Tensor: output features of the same shape as the input.
        """
        b, n, c = x.shape

        x = rearrange(x, "b n c -> n (b c)")
        x = knn_interpolate(x, self.pos_x, self.pos_y)
        x = rearrange(x, "(h w) (b c) -> b c h w", b=b, c=c, h=self.i_h, w=self.i_w)
        x = self.image_meta_model(x)

        x = rearrange(x, "b c h w -> (h w) (b c)")
        x = knn_interpolate(x, self.pos_y, self.pos_x)
        x = rearrange(x, "n (b c) -> b n c", b=b, c=c)
        return x


class WrapperMetaModel(nn.Module):
    """Reuse a trained MetaModel on lat/lon data of a higher resolution.

    The points are interpolated onto a grid scale_factor times finer than the one of the
    wrapped model, that grid is split into windows stacked along the batch dimension, and
    the weights of the wrapped model are loaded into a new one whose transformer also
    attends across the windows.
    """

    def __init__(self, lat_lons: list, meta_model: MetaModel, scale_factor):
        """Initialize WrapperMetaModel.

        Args:
            lat_lons (list): latitude and longitude of each point of the input.
            meta_model (MetaModel): model whose settings and weights are reused.
            scale_factor (int or tuple): number of windows along the rows and along the
                columns.
        """
        super().__init__()
        s_h, s_w = pair(scale_factor)
        self.i_h, self.i_w = meta_model.i_h * s_h, meta_model.i_w * s_w
        self.pos_x = torch.tensor(lat_lons)
        self.pos_y = torch.cartesian_prod(
            (torch.arange(-self.i_h / 2, self.i_h / 2, 1) / self.i_h * 180).to(torch.long),
            (torch.arange(0, self.i_w, 1) / self.i_w * 360).to(torch.long),
        )

        self.batcher = Rearrange("b c (h s_h) (w s_w) -> (b s_h s_w) c h w", s_h=s_h, s_w=s_w)

        imm_args = vars(meta_model.image_meta_model)
        imm_args.update({"res": True, "scale_factor": scale_factor})
        self.image_meta_model = ImageMetaModel(**imm_args)
        self.image_meta_model.load_state_dict(
            meta_model.image_meta_model.state_dict(), strict=False
        )

        self.debatcher = Rearrange("(b s_h s_w) c h w -> b c (h s_h) (w s_w)", s_h=s_h, s_w=s_w)

    def forward(self, x):
        """Interpolate onto the finer grid, run the wrapped model and interpolate back.

        Args:
            x (torch.Tensor): input features of shape (batch, num_points, channels).

        Returns:
            torch.Tensor: output features of the same shape as the input.
        """
        b, n, c = x.shape

        x = rearrange(x, "b n c -> n (b c)")
        x = knn_interpolate(x, self.pos_x, self.pos_y)
        x = rearrange(x, "(h w) (b c) -> b c h w", b=b, c=c, h=self.i_h, w=self.i_w)

        x = self.batcher(x)
        x = self.image_meta_model(x)
        x = self.debatcher(x)

        x = rearrange(x, "b c h w -> (h w) (b c)")
        x = knn_interpolate(x, self.pos_y, self.pos_x)
        x = rearrange(x, "n (b c) -> b n c", b=b, c=c)

        return x


class LoRALayer(nn.Module):
    """Linear layer paired with a trainable low-rank update.

    The output of the wrapped layer is summed with B @ A @ x, where A and B are the two
    factors of a rank r matrix. B starts at zero, so the layer initially behaves exactly
    like the one it wraps.
    """

    def __init__(self, linear_layer: nn.Module, r: int):
        """
        Initialize LoRALayer.

        Args:
            linear_layer (nn.Module): Linear layer to be transformed.
            r (int): rank of the low-rank matrix.
        """
        super().__init__()
        out_features, in_features = linear_layer.weight.shape

        self.A = nn.Parameter(torch.randn(r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, r))
        self.linear_layer = linear_layer

    def forward(self, x):
        """Sum the output of the wrapped layer with the low-rank update.

        Args:
            x (torch.Tensor): input tensor accepted by the wrapped linear layer.

        Returns:
            torch.Tensor: output of the wrapped layer plus the low-rank correction.
        """
        out = self.linear_layer(x) + self.B @ self.A @ x
        return out


class LoRAModule(nn.Module):
    """Model whose linear layers are replaced by their LoRALayer counterparts.

    Every submodule is put in evaluation mode before each nn.Linear is wrapped, so that
    the low-rank factors are the only parameters left to train.
    """

    def __init__(self, model, r=4):
        """
        Initialize LoRAModule.

        Args:
            model (nn.Module): Model to be modified with LoRA layers.
            r (int, optional): Rank of LoRA layers. Defaults to 4.
        """
        super().__init__()
        for name, layer in model.named_modules():
            layer.eval()
            if isinstance(layer, nn.Linear):
                lora_layer = LoRALayer(layer, r)
                setattr(model, name, lora_layer)
        self.model = model

    def forward(self, x):
        """Run the modified model on the input.

        Args:
            x (torch.Tensor): input tensor accepted by the wrapped model.

        Returns:
            torch.Tensor: output of the wrapped model.
        """
        return self.model(x)
