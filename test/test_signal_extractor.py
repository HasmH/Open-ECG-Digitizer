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


def _stacked_bands(centers: list[int], height: int, width: int, spike: tuple[int, int, int] | None = None) -> torch.Tensor:
    """Build a synthetic stacked single-column layout: one flat trace per band.

    ``spike`` optionally adds a tall upward deflection ``(band_index, x, amplitude)``
    to one band, mimicking an R wave that reaches into the neighbouring band.
    """
    fmap = torch.zeros(height, width)
    xs = np.arange(width)
    for i, center in enumerate(centers):
        ys = np.full(width, float(center))
        if spike is not None and spike[0] == i:
            _, x0, amplitude = spike
            ys = ys - np.exp(-((xs - x0) ** 2) / (2 * 20.0**2)) * amplitude
        for x in range(width):
            y = int(round(ys[x]))
            fmap[max(0, y - 2) : min(height, y + 3), x] = 1.0
    return fmap


def test_banded_extraction_returns_one_line_per_band() -> None:
    centers = [50, 150, 250]
    fmap = _stacked_bands(centers, height=300, width=200)
    extractor = SignalExtractor(num_bands=3, min_line_width=5)

    assert extractor._find_band_centers(fmap, 3) == centers

    lines = extractor(fmap)
    assert lines.shape[0] == 3
    baselines = torch.nanmedian(lines, dim=1).values
    assert torch.allclose(baselines, torch.tensor([50.0, 150.0, 250.0]), atol=3.0)


def test_banded_extraction_does_not_clip_tall_peaks() -> None:
    # Middle band has an R wave reaching 70 px up, well past the 50 px half-band
    # boundary that a hard band cut would clip it at.
    centers = [50, 150, 250]
    fmap = _stacked_bands(centers, height=300, width=200, spike=(1, 100, 70))
    extractor = SignalExtractor(num_bands=3, min_line_width=5)

    middle = extractor(fmap)[1]
    peak_y = float(middle.nan_to_num(float("inf")).min())
    assert peak_y < 95.0, f"peak clipped at y={peak_y}, expected it to reach ~80"
