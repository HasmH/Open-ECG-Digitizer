import matplotlib.pyplot as plt
import numpy as np
import torch

from src.digitize import plot_extracted_trace_overlay


def test_plot_extracted_trace_overlay_uses_pixel_coordinates() -> None:
    raw_lines = torch.tensor([[1.0, 2.0, float("nan")], [4.0, 5.0, 6.0]])
    fig, ax = plt.subplots()

    plot_extracted_trace_overlay(ax, raw_lines)

    assert len(ax.lines) == 2
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [0, 1, 2])
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [1.0, 2.0, np.nan])
    assert ax.lines[0].get_label() == "Extracted centreline"
    assert ax.lines[1].get_label().startswith("_")
    plt.close(fig)


def test_plot_extracted_trace_overlay_accepts_no_lines() -> None:
    fig, ax = plt.subplots()

    plot_extracted_trace_overlay(ax, torch.empty((0, 10)))

    assert not ax.lines
    assert ax.get_legend() is None
    plt.close(fig)
