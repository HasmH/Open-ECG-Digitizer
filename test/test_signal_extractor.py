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


def test_random_seed_makes_path_tie_breaking_repeatable() -> None:
    extractor = SignalExtractor(random_seed=42)
    image = torch.zeros((5, 2), dtype=torch.float32)
    candidates = torch.arange(5)

    extractor._reset_random_generator()
    first = extractor._get_pixel_vals(image, candidates, x=1)
    extractor._reset_random_generator()
    second = extractor._get_pixel_vals(image, candidates, x=1)

    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_unseeded_path_tie_breaking_remains_stochastic() -> None:
    extractor = SignalExtractor()
    image = torch.zeros((5, 2), dtype=torch.float32)
    candidates = torch.arange(5)

    first = extractor._get_pixel_vals(image, candidates, x=1)
    second = extractor._get_pixel_vals(image, candidates, x=1)

    assert not torch.equal(first, second)
