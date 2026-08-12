"""
aurora/model.py - Core implementation of Aurora model for unstructured point data
"""

from typing import Optional

import torch
import torch.nn as nn


class PointEncoder(nn.Module):
    """Embed unstructured points and their features into a shared latent space.

    Coordinates and features are embedded by two separate MLPs and then added together, so
    no positional embedding is used and the encoding of a point does not depend on its
    position in the sequence.
    """

    def __init__(self, input_features: int, embed_dim: int, max_seq_len: int = 1024):
        """Build the coordinate and feature embeddings.

        Args:
            input_features (int): number of feature channels attached to each point.
            embed_dim (int): dimension of the embedding produced for each point.
            max_seq_len (int): maximum number of points kept by the forward pass; points
                beyond this count are dropped. Defaults to 1024.
        """
        super().__init__()
        self.input_dim = input_features + 2  # Account for lat/lon coordinates
        self.max_seq_len = max_seq_len

        # Remove positional embeddings as they break point ordering invariance

        # Enhanced coordinate embedding
        self.coord_encoder = nn.Sequential(
            nn.Linear(2, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )

        # Feature embedding
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Final normalization
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, points: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Encode points and their features into a single embedding per point.

        Coordinates are rescaled to the [-1, 1] range by dividing the longitude by 180 and
        the latitude by 90. Both tensors are truncated to max_seq_len points.

        Args:
            points (torch.Tensor): coordinates of shape (batch_size, num_points, 2), holding
                the longitude in degrees at index 0 and the latitude in degrees at index 1.
            features (torch.Tensor): features of shape (batch_size, num_points,
                input_features).

        Returns:
            torch.Tensor: embeddings of shape (batch_size, kept_points, embed_dim), where
                kept_points is num_points capped at max_seq_len.
        """
        num_points = points.shape[1]
        if num_points > self.max_seq_len:
            points = points[:, : self.max_seq_len, :]
            features = features[:, : self.max_seq_len, :]

        # Normalize coordinates to [-1, 1] range
        normalized_points = torch.stack(
            [points[..., 0] / 180.0, points[..., 1] / 90.0],
            dim=-1,  # longitude  # latitude
        )

        # Separately encode coordinates and features
        coord_embedding = self.coord_encoder(normalized_points)
        feature_embedding = self.feature_encoder(features)

        # Combine embeddings through addition (order-invariant operation)
        x = coord_embedding + feature_embedding

        # Final normalization
        x = self.norm(x)

        return x


class PointDecoder(nn.Module):
    """Decodes latent representations back to point features."""

    def __init__(self, embed_dim: int, output_features: int):
        """Build the decoding MLP.

        Args:
            embed_dim (int): dimension of the latent representation of each point.
            output_features (int): number of feature channels predicted for each point.
        """
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, output_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode latent representations into point features.

        Args:
            x (torch.Tensor): (batch_size, num_points, embed_dim) tensor

        Returns:
            (batch_size, num_points, output_features) tensor
        """
        return self.decoder(x)


class PointCloudProcessor(nn.Module):
    """Processes point cloud data using self-attention layers."""

    def __init__(self, embed_dim: int, num_layers: int = 4):
        """Stack the self-attention layers.

        Args:
            embed_dim (int): dimension of the latent representation of each point.
            num_layers (int): number of self-attention layers to apply. Defaults to 4.
        """
        super().__init__()
        self.layers = nn.ModuleList([SelfAttentionLayer(embed_dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply every self-attention layer in turn.

        Args:
            x (torch.Tensor): (batch_size, num_points, embed_dim) tensor

        Returns:
            (batch_size, num_points, embed_dim) tensor after processing
        """
        for layer in self.layers:
            x = layer(x)
        return x


class SelfAttentionLayer(nn.Module):
    """Transformer block made of multi-head self-attention and a feed-forward network.

    Each of the two sub-layers is wrapped in a residual connection which is followed by a
    layer normalization.
    """

    def __init__(self, embed_dim: int):
        """Build the attention block, the normalizations and the feed-forward network.

        Args:
            embed_dim (int): dimension of the latent representation of each point. The
                attention uses 8 heads and the feed-forward network expands the dimension
                by a factor of 4.
        """
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim), nn.ReLU(), nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run self-attention and the feed-forward network over the points.

        The input is transposed to the sequence-first layout expected by the attention
        module and transposed back before the residual connection is applied.

        Args:
            x (torch.Tensor): (batch_size, num_points, embed_dim) tensor

        Returns:
            torch.Tensor: (batch_size, num_points, embed_dim) tensor
        """
        # First attention block with residual
        x_t = x.transpose(0, 1)
        attended, _ = self.attention(x_t, x_t, x_t)
        attended = attended.transpose(0, 1)
        x = self.norm1(x + attended)

        # FFN block with residual
        x = self.norm2(x + self.ffn(x))
        return x


class EarthSystemLoss(nn.Module):
    """Weighted sum of a mean squared error, a spatial correlation and a physical term."""

    def __init__(self, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2):
        """Store the weights of the three loss terms.

        Args:
            alpha (float): weight of the mean squared error term. Defaults to 0.5.
            beta (float): weight of the spatial correlation term. Defaults to 0.3.
            gamma (float): weight of the physical consistency term. Defaults to 0.2.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def spatial_correlation_loss(
        self, pred: torch.Tensor, target: torch.Tensor, points: torch.Tensor
    ) -> torch.Tensor:
        """Penalize differences between neighbouring points that the target does not show.

        Pairwise distances are computed between the point coordinates and only the pairs
        that lie closer than 5 degrees are kept. For those pairs the difference between two
        predictions is compared with the difference between the two matching targets.

        Args:
            pred (torch.Tensor): predictions of shape (batch_size, num_points, features).
            target (torch.Tensor): targets of shape (batch_size, num_points, features).
            points (torch.Tensor): coordinates of shape (batch_size, num_points, 2).

        Returns:
            torch.Tensor: scalar tensor with the mean squared difference over nearby pairs.
        """
        batch_size, num_points, _ = points.shape
        points_flat = points.view(-1, 2)

        # Compute pairwise distances
        dists = torch.cdist(points_flat, points_flat)
        dists = dists.view(batch_size, num_points, num_points)

        # Create mask for nearby points (5 degrees threshold)
        nearby_mask = (dists < 5.0).float().unsqueeze(-1)

        # Compute differences
        pred_diff = pred.unsqueeze(2) - pred.unsqueeze(1)
        target_diff = target.unsqueeze(2) - target.unsqueeze(1)

        # Calculate loss with proper broadcasting
        correlation_loss = torch.mean(nearby_mask * (pred_diff - target_diff).pow(2))

        return correlation_loss

    def physical_loss(self, pred: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """Calculate physical consistency loss - ensures predictions follow basic physical laws"""
        # Ensure non-negative values for physical quantities (e.g., temperature in Kelvin)
        min_value_loss = torch.nn.functional.relu(-pred).mean()

        # Ensure reasonable maximum values (e.g., max temperature)
        max_value_loss = torch.nn.functional.relu(pred - 500).mean()  # Assuming max value of 500

        # Add latitude-based consistency (e.g., colder at poles)
        latitude = points[..., 1]  # Second coordinate is latitude
        abs_latitude = torch.abs(latitude)
        latitude_consistency = torch.mean(
            torch.nn.functional.relu(pred[..., 0] - (1.0 - abs_latitude / 90.0) * pred.mean())
        )

        # Combine physical constraints
        physical_loss = min_value_loss + max_value_loss + 0.1 * latitude_consistency
        return physical_loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor, points: torch.Tensor) -> dict:
        """Combine the mean squared error, spatial and physical terms.

        Args:
            pred (torch.Tensor): predictions of shape (batch_size, num_points, features).
            target (torch.Tensor): targets of shape (batch_size, num_points, features).
            points (torch.Tensor): coordinates of shape (batch_size, num_points, 2).

        Returns:
            dict: the weighted sum under the key "total_loss", together with the three
                unweighted terms under the keys "mse_loss", "spatial_correlation_loss" and
                "physical_loss".
        """
        mse_loss = torch.nn.functional.mse_loss(pred, target)
        spatial_loss = self.spatial_correlation_loss(pred, target, points)
        physical_loss = self.physical_loss(pred, points)

        # Combine losses with the specified weights
        total_loss = self.alpha * mse_loss + self.beta * spatial_loss + self.gamma * physical_loss

        return {
            "total_loss": total_loss,
            "mse_loss": mse_loss,
            "spatial_correlation_loss": spatial_loss,
            "physical_loss": physical_loss,
        }


class AuroraModel(nn.Module):
    """Encoder, processor and decoder pipeline for unstructured point data.

    Points and their features are embedded by a PointEncoder, refined by a
    PointCloudProcessor and mapped back to features by a PointDecoder.
    """

    def __init__(
        self,
        input_features: int,
        output_features: int,
        latent_dim: int = 256,
        num_layers: int = 4,
        max_points: int = 10000,
        max_seq_len: int = 1024,
        use_checkpointing: bool = False,
    ):
        """Build the encoder, the processor and the decoder, then initialize the weights.

        Args:
            input_features (int): number of feature channels attached to each input point.
            output_features (int): number of feature channels predicted for each point.
            latent_dim (int): dimension of the latent representation. Defaults to 256.
            num_layers (int): number of self-attention layers in the processor. Defaults
                to 4.
            max_points (int): number of points above which the forward pass raises a
                ValueError. Defaults to 10000.
            max_seq_len (int): number of points above which the encoder truncates its
                input. Defaults to 1024.
            use_checkpointing (bool): if true, gradient checkpointing is applied to the
                processor during training. Defaults to False.
        """
        super().__init__()

        self.max_points = max_points
        self.max_seq_len = max_seq_len
        self.input_features = input_features
        self.output_features = output_features

        # Model components
        self.encoder = PointEncoder(input_features, latent_dim, max_seq_len)
        self.processor = PointCloudProcessor(latent_dim, num_layers)
        self.decoder = PointDecoder(latent_dim, output_features)

        # Add gradient checkpointing
        self.use_checkpointing = use_checkpointing

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, points: torch.Tensor, features: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Predict the output features of every point.

        Args:
            points (torch.Tensor): coordinates of shape (batch_size, num_points, 2), holding
                the longitude in degrees at index 0 and the latitude in degrees at index 1.
            features (torch.Tensor): features of shape (batch_size, num_points,
                input_features).
            mask (Optional[torch.Tensor]): mask of shape (batch_size, num_points) applied to
                the inputs and to the output. Defaults to None.

        Returns:
            torch.Tensor: predictions of shape (batch_size, num_points, output_features).

        Raises:
            ValueError: if the number of points is larger than max_points.
        """
        if points.shape[1] > self.max_points:
            raise ValueError(
                f"Number of points ({points.shape[1]}) exceeds maximum ({self.max_points})"
            )

        # Handle mask properly
        if mask is not None:
            mask = mask.float().unsqueeze(-1)
            points = points * mask
            features = features * mask

        # Forward pass with gradient checkpointing
        x = self.encoder(points, features)

        if self.use_checkpointing and self.training:
            x = torch.utils.checkpoint.checkpoint(self.processor, x)
        else:
            x = self.processor(x)

        output = self.decoder(x)

        # Apply mask to output if provided
        if mask is not None:
            output = output * mask

        return output
