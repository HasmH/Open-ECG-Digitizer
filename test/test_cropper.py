from types import SimpleNamespace

import torch

from src.model.cropper import Cropper
from src.model.inference_wrapper import InferenceWrapper


def test_vertical_padding_extends_skewed_source_points() -> None:
    cropper = Cropper(vertical_padding_fraction=0.1)
    source_points = torch.tensor(
        [
            [10.0, 20.0],
            [90.0, 10.0],
            [95.0, 80.0],
            [5.0, 90.0],
        ]
    )

    padded = cropper._add_vertical_padding(source_points, height=100, width=100)

    expected = torch.tensor(
        [
            [10.5, 13.0],
            [89.5, 3.0],
            [95.5, 87.0],
            [4.5, 97.0],
        ]
    )
    assert torch.allclose(padded, expected)


def test_vertical_padding_is_clamped_to_image_bounds() -> None:
    cropper = Cropper(vertical_padding_fraction=0.2)
    source_points = torch.tensor(
        [
            [0.0, 1.0],
            [99.0, 1.0],
            [99.0, 98.0],
            [0.0, 98.0],
        ]
    )

    padded = cropper._add_vertical_padding(source_points, height=100, width=100)

    assert torch.equal(padded[:, 1], torch.tensor([0.0, 0.0, 99.0, 99.0]))


def test_post_alignment_crop_retains_vertical_padding() -> None:
    wrapper = InferenceWrapper.__new__(InferenceWrapper)
    wrapper.cropper = SimpleNamespace(vertical_padding_fraction=0.1)
    image = torch.zeros((1, 3, 100, 20))
    signal_prob = torch.zeros((1, 1, 100, 20))
    grid_prob = torch.zeros_like(signal_prob)
    text_prob = torch.zeros_like(signal_prob)
    signal_prob[:, :, 20:80, :] = 1

    cropped = wrapper._crop_y(image, signal_prob, grid_prob, text_prob)

    assert all(tensor.shape[-2] == 72 for tensor in cropped)
