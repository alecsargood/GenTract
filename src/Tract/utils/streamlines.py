import math

import torch


def generate_rotation_matrix_torch(axis: int, angle_deg: float, device: torch.device) -> torch.Tensor:
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    matrix = torch.eye(4, dtype=torch.float32, device=device)

    if axis == 0:
        matrix[1, 1] = cos_a
        matrix[1, 2] = -sin_a
        matrix[2, 1] = sin_a
        matrix[2, 2] = cos_a
    elif axis == 1:
        matrix[0, 0] = cos_a
        matrix[0, 2] = sin_a
        matrix[2, 0] = -sin_a
        matrix[2, 2] = cos_a
    elif axis == 2:
        matrix[0, 0] = cos_a
        matrix[0, 1] = -sin_a
        matrix[1, 0] = sin_a
        matrix[1, 1] = cos_a
    else:
        raise ValueError("Axis must be 0, 1, or 2")
    return matrix


def apply_affine_transform_torch(coords: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    if coords.numel() == 0:
        return coords

    original_shape = coords.shape
    coords_flat = coords.view(-1, 3)
    rotation_matrix = matrix[:3, :3]
    transformed_flat = coords_flat @ rotation_matrix
    return transformed_flat.view(original_shape)
