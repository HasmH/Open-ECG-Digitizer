import numpy as np
import torch

from src.model.signal_extractor import SignalExtractor


def test_height_difference_penalty_is_configurable() -> None:
    min_coords = torch.tensor([[10.0, 5.0], [20.0, 20.0]])
    max_coords = torch.tensor([[0.0, 5.0], [5.0, 20.0]])
    heights = torch.tensor([5.0, 20.0])

    unpenalized, _ = SignalExtractor(height_difference_penalty=0).compute_cost_matrix(
        min_coords, max_coords, W=100, heights=heights
    )
    penalized, _ = SignalExtractor(height_difference_penalty=30).compute_cost_matrix(
        min_coords, max_coords, W=100, heights=heights
    )

    assert penalized[0, 1] > unpenalized[0, 1]
    np.testing.assert_allclose(penalized.diagonal(), unpenalized.diagonal())


def test_height_difference_penalty_rejects_negative_values() -> None:
    try:
        SignalExtractor(height_difference_penalty=-1)
    except ValueError as error:
        assert str(error) == "height_difference_penalty must be non-negative"
    else:
        raise AssertionError("Expected a negative height_difference_penalty to be rejected")
