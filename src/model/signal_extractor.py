from collections import defaultdict, deque
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import torchvision
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks
from skimage.measure import label


class SignalExtractor:
    threshold_sum: float
    threshold_line_in_mask: float
    label_thresh: float
    max_iterations: int
    split_num_stripes: int
    candidate_span: int
    debug: int
    lam: float
    min_line_width: int
    height_difference_penalty: float
    num_peaks: Optional[int]
    x_offset: int
    num_bands: Optional[int]
    band_overlap_fraction: float
    band_smoothing_fraction: float

    def __init__(
        self,
        threshold_sum: float = 10.0,
        threshold_line_in_mask: float = 0.95,
        label_thresh: float = 0.1,
        max_iterations: int = 4,
        split_num_stripes: int = 4,
        candidate_span: int = 10,
        debug: int = 0,
        lam: float = 0.5,
        min_line_width: int = 30,
        height_difference_penalty: float = 30.0,
        num_bands: Optional[int] = None,
        band_overlap_fraction: float = 0.35,
        band_smoothing_fraction: float = 0.1,
    ) -> None:
        if height_difference_penalty < 0:
            raise ValueError("height_difference_penalty must be non-negative")
        if num_bands is not None and num_bands < 1:
            raise ValueError("num_bands must be a positive integer or None")
        self.threshold_sum = threshold_sum
        self.threshold_line_in_mask = threshold_line_in_mask
        self.label_thresh = label_thresh
        self.max_iterations = max_iterations
        self.split_num_stripes = split_num_stripes
        self.candidate_span = candidate_span
        self.debug = debug
        self.lam = lam
        self.min_line_width = min_line_width
        self.height_difference_penalty = height_difference_penalty
        self.num_bands = num_bands
        self.band_overlap_fraction = band_overlap_fraction
        self.band_smoothing_fraction = band_smoothing_fraction
        self.num_peaks = None
        self.x_offset = 0

    def __call__(self, feature_map: torch.Tensor) -> torch.Tensor:
        fmap = feature_map.cpu().clone()
        if self.num_bands is not None and self.num_bands > 1:
            return self._banded_extract_and_merge(fmap, feature_map.shape[1])
        lines_list = self._iterative_extraction(fmap)
        self.num_peaks = self._autodetect_num_peaks(fmap)
        lines_list = [ln for ln in lines_list if (~torch.isnan(ln)).sum() > self.min_line_width]
        if len(lines_list) == 0:
            return torch.empty((0, feature_map.shape[1]), dtype=torch.float32)
        lines = torch.stack(lines_list, dim=0)
        merged_lines_list, overlaps = self.match_and_merge_lines(lines)
        if len(merged_lines_list) == 0:
            return torch.empty((0, feature_map.shape[1]), dtype=torch.float32)
        merged_lines = torch.stack(merged_lines_list, dim=0)
        if self.num_peaks != len(merged_lines):
            print(
                f"Warning: Number of peaks ({self.num_peaks}) does not match number of merged lines ({len(merged_lines)})."
            )
        return merged_lines

    def _iterative_extraction(self, fmap: torch.Tensor) -> list[torch.Tensor]:
        for it in range(self.max_iterations):
            good, rejected, rej_maps = self._extract_candidate_lines(fmap)
            if not rejected:
                break
            self._refine_fmap_by_removal(fmap, rej_maps)
        return good

    def _banded_extract_and_merge(self, fmap: torch.Tensor, width: int) -> torch.Tensor:
        """Extract and merge one line per lead in a stacked single-column layout.

        For layouts such as 12x1 the leads occupy known, roughly evenly spaced
        horizontal bands centred on their baselines. Each lead is extracted from
        a window centred on its baseline that overlaps into the neighbouring
        bands, so a tall deflection is captured in full instead of being clipped
        at a hard band boundary. Fragments whose baseline does not belong to the
        lead (signal that leaked in from a neighbour) are discarded, and the
        remainder are merged into a single line -- guaranteeing one line per
        lead without relying on the global fragment matcher.

        Returns:
            A ``(num_bands, W)`` tensor of merged lines, top band first, with
            NaN where no signal was recovered.
        """
        self.num_peaks = self._autodetect_num_peaks(fmap)
        centers = self._find_band_centers(fmap, self.num_bands)  # type: ignore[arg-type]
        if self.debug:
            self._plot_band_centers(fmap, centers)
        if not centers:
            return torch.empty((0, width), dtype=torch.float32)

        H = fmap.shape[0]
        band_height = self._median_spacing(centers, H)
        reach = max(1, int(band_height * (0.5 + self.band_overlap_fraction)))

        band_lines = []
        for center in centers:
            y0, y1 = max(0, center - reach), min(H, center + reach)
            strip = fmap[y0:y1, :].clone()
            band_lines.append(self._extract_band_line(strip, y0, center, band_height, width))

        lines = self.preprocess_lines(torch.stack(band_lines, dim=0))
        if self.debug:
            self.plot_lines(lines, "Preprocessed Lines")
        return lines

    def _extract_band_line(
        self, strip: torch.Tensor, y0: int, center: int, band_height: int, width: int
    ) -> torch.Tensor:
        """Extract one merged line for the lead centred at ``center``.

        A fragment is kept only when its baseline (median row) lies within half
        a band of ``center`` -- i.e. it belongs to this lead. Fragments further
        out are a neighbouring lead's trace that leaked into the overlap margin;
        filling this lead's honest gaps with them just injects the neighbour's
        deflection spikes as noise, so they are dropped. The survivors (this
        lead's own baseline and deflection pieces) are combined by priority: the
        longest forms the line, and shorter ones only fill columns it left NaN.
        """
        fragments = [line + float(y0) for line in self._iterative_extraction(strip)]
        fragments = [f for f in fragments if int((~torch.isnan(f)).sum()) > self.min_line_width]
        fragments = [f for f in fragments if abs(float(torch.nanmedian(f)) - center) < band_height * 0.5]
        if not fragments:
            return torch.full((width,), float("nan"))

        fragments.sort(key=lambda f: int((~torch.isnan(f)).sum()), reverse=True)
        merged = fragments[0].clone()
        for fragment in fragments[1:]:
            fill = torch.isnan(merged) & ~torch.isnan(fragment)
            merged[fill] = fragment[fill]
        return merged

    def _find_band_centers(self, fmap: torch.Tensor, num_bands: int) -> list[int]:
        """Locate the baseline row of each stacked lead.

        Baselines (flat isoelectric segments) span the full width and form the
        sharpest, most reliable peaks in the row-mass profile. If peak detection
        is unreliable, fall back to evenly spaced centres across the active span.
        """
        H = fmap.shape[0]
        row_mass = self._smooth_row_mass(
            fmap.sum(dim=1), max(1, int(H / num_bands * self.band_smoothing_fraction))
        )
        row_mass_np = row_mass.numpy()

        peaks, props = find_peaks(
            row_mass_np,
            distance=max(1, H // (2 * num_bands)),
            prominence=float(row_mass_np.max()) * 0.03,
        )
        if len(peaks) >= num_bands:
            strongest = np.argsort(props["prominences"])[::-1][:num_bands]
            return sorted(int(p) for p in peaks[strongest])

        return self._even_band_centers(row_mass, num_bands)

    def _even_band_centers(self, row_mass: torch.Tensor, num_bands: int) -> list[int]:
        H = row_mass.shape[0]
        active = (row_mass > row_mass.max() * 0.02).nonzero()
        if active.numel() == 0:
            y_top, y_bot = 0, H - 1
        else:
            y_top, y_bot = int(active[0].item()), int(active[-1].item())
        band_height = max(1, (y_bot - y_top) / num_bands)
        return [int(y_top + band_height * (i + 0.5)) for i in range(num_bands)]

    def _median_spacing(self, centers: list[int], fallback: int) -> int:
        if len(centers) < 2:
            return fallback
        return int(np.median(np.diff(sorted(centers))))

    def _plot_band_centers(self, fmap: torch.Tensor, centers: list[int]) -> None:
        row_mass = fmap.sum(dim=1)
        band_height = self._median_spacing(centers, fmap.shape[0])
        reach = max(1, int(band_height * (0.5 + self.band_overlap_fraction)))
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
        ax0.imshow(fmap.numpy(), aspect="auto", cmap="magma")
        for c in centers:
            ax0.axhline(c, color="lime", linewidth=0.8)
            ax0.axhspan(c - reach, c + reach, color="cyan", alpha=0.06)
        ax0.set_title("signal_prob: baselines (green) and overlapping bands")
        ax1.plot(row_mass.numpy(), np.arange(len(row_mass)))
        for c in centers:
            ax1.axhline(c, color="lime", linewidth=0.8)
        ax1.invert_yaxis()
        ax1.set_title("row mass (sum over x)")
        plt.savefig("sandbox/band_boundaries.png", dpi=110)
        plt.close()

    def _smooth_row_mass(self, row_mass: torch.Tensor, kernel_size: int) -> torch.Tensor:
        if kernel_size <= 1:
            return row_mass
        pad = kernel_size // 2
        kernel = torch.ones(1, 1, kernel_size) / kernel_size
        padded = torch.nn.functional.pad(row_mass.view(1, 1, -1), (pad, pad), mode="replicate")
        return torch.nn.functional.conv1d(padded, kernel).view(-1)

    def _extract_candidate_lines(
        self, fmap: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        lab: npt.NDArray[Any] = self._label_regions(fmap)
        zero_mask = lab == 0
        big_offset = lab.max() + 1000  # Just some big number to avoid overlap

        for i in range(self.split_num_stripes):
            sl = slice(
                i * lab.shape[1] // self.split_num_stripes,
                (i + 1) * lab.shape[1] // self.split_num_stripes,
            )
            relab: npt.NDArray[np.int64] = label(lab[:, sl] > 0, connectivity=1)  # type: ignore
            lab[:, sl] += big_offset * i + relab
        lab[zero_mask] = 0

        good: list[torch.Tensor] = []
        rejected: list[torch.Tensor] = []
        rej_maps: list[torch.Tensor] = []
        for lid in np.unique(lab):
            if lid == 0:
                continue
            mask = torch.tensor(lab == lid)
            line = self._extract_line_from_region(fmap, mask)
            if self._classify_line(line, mask):
                good.append(line)
            else:
                rejected.append(line)
                rej_maps.append(fmap * mask)

        for line in (*good, *rejected):
            line[line < 5] = float("nan")

        return good, rejected, rej_maps

    def _refine_fmap_by_removal(self, fmap: torch.Tensor, rej_maps: list[torch.Tensor]) -> None:
        for i, mask_map in enumerate(rej_maps):
            ys, xs = torch.where(mask_map > 0)
            if ys.numel() == 0:
                continue
            y0, y1 = int(ys.min().item()), int(ys.max().item())
            x0, x1 = int(xs.min().item()), int(xs.max().item())
            cropped: npt.NDArray[Any] = mask_map[y0 : y1 + 1, x0 : x1 + 1].cpu().numpy()
            _, new_img = self._trace_horizontal_path(cropped)
            region = fmap[y0 : y1 + 1, x0 : x1 + 1]
            region[cropped > 0] = new_img[cropped > 0]

    def _label_regions(self, fmap: torch.Tensor) -> npt.NDArray[np.int64]:
        lab: npt.NDArray[np.int64] = label((fmap.numpy() > self.label_thresh).astype(int), connectivity=1)  # type: ignore
        for lb in np.unique(lab):
            if lb == 0:
                continue
            if fmap[lab == lb].sum() < self.threshold_sum:
                lab[lab == lb] = 0
        output: npt.NDArray[np.int64] = label(lab > 0, connectivity=1)  # type: ignore
        return output

    def _extract_line_from_region(self, fmap: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(fmap.shape[0]).view(-1, 1)
        masked = torch.where(mask, fmap, torch.zeros_like(fmap))
        masked /= masked.sum(0, keepdim=True).clamp(min=1e-6)
        line = (masked * pos).sum(0)
        line[line < 1 / fmap.shape[0]] = float("nan")
        return line

    def _classify_line(self, line: torch.Tensor, mask: torch.Tensor) -> bool:
        lf = line.long().clamp(0, mask.shape[0] - 1)
        cols = mask.any(0)
        in_mask = mask[lf, torch.arange(mask.shape[1])]
        return bool(in_mask[cols].float().mean().item() >= self.threshold_line_in_mask)

    def _trace_horizontal_path(self, img_arr: npt.NDArray[Any]) -> tuple[torch.Tensor, torch.Tensor]:
        blurry = torch.tensor(img_arr, dtype=torch.float32)
        blurry += (torch.linspace(-1, 1, blurry.shape[0]) ** 6).unsqueeze(1)
        img = torch.tensor(img_arr, dtype=torch.float32)
        H, W = img.shape
        path_y: list[int] = [H // 2]

        for x in range(1, W):
            prev = path_y[-1]
            candidates = torch.arange(prev - self.candidate_span, prev + self.candidate_span + 1).clamp(0, H - 1)
            vals = self._get_pixel_vals(blurry, candidates, x)
            path_y.append(int(candidates[torch.argmin(vals)].item()))

        for i in range(len(path_y) - 1):
            lo, hi = sorted((path_y[i], path_y[i + 1]))
            img[lo - 1 : hi + 2, i] = 0

        return torch.tensor(path_y), img

    def _get_pixel_vals(self, blurry_img: torch.Tensor, candidates: torch.Tensor, x: int) -> torch.Tensor:
        middle = len(candidates) // 2
        this_col = blurry_img[candidates, x - 1]
        other_col = blurry_img[candidates, x]
        pixel_vals: list[torch.Tensor] = []
        for i in range(len(candidates)):
            seg = slice(min(i, middle), max(i, middle) + 1)
            pixel_vals.append((this_col[seg].mean() + other_col[seg].mean()) / 2)
        vals = torch.stack(pixel_vals)
        vals += torch.randn(len(candidates)) * 1e-6
        vals += vals.std() * torch.tensor(np.linspace(-1, 1, len(candidates)) ** 2, dtype=torch.float32)
        return vals

    def _blur(self, tensor: torch.Tensor, kernel_size: int = 3, sigma: float = 1.0) -> torch.Tensor:
        """
        Applies a Gaussian blur to the input tensor.

        Args:
            tensor (torch.Tensor): The input tensor to be blurred.
            kernel_size (int): The size of the Gaussian kernel.

        Returns:
            torch.Tensor: The blurred tensor.
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = torchvision.transforms.GaussianBlur(kernel_size, sigma=sigma)
        blurred_tensor: torch.Tensor = kernel(tensor.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        return blurred_tensor

    def _autodetect_num_peaks(self, fmap: torch.Tensor) -> int:
        logits = torch.log(1 - fmap.clamp(1e-6, 1 - 1e-6)) / torch.log(fmap.clamp(1e-6, 1 - 1e-6))
        logits = self._blur(logits)
        probs_raw = torch.softmax(logits, dim=0)
        probs = (probs_raw - 1.1 / probs_raw.shape[0]).clamp(0)

        zero_mask = probs == 0
        nonzero_mask = ~zero_mask
        lines = zero_mask[1:] & nonzero_mask[:-1]
        line_counts = lines.sum(0)
        value_counts = torch.bincount(line_counts)
        most_common_value = int(value_counts.argmax().item())
        most_common_count = int(value_counts.max().item())

        if most_common_count < line_counts.shape[0] * 0.5:
            print(
                f"Warning: Autodetected number of peaks ({most_common_value}) is not reliable, "
                f"consider setting it manually or adjusting the autodetection parameters."
            )

        return most_common_value

    def preprocess_lines(self, lines: torch.Tensor) -> torch.Tensor:
        lines = lines.clone()
        lines[lines == 0] = float("nan")
        valid_cols = lines.nan_to_num(0.0).abs().sum(0) > 0
        first, last = torch.nonzero(valid_cols, as_tuple=True)[0][[0, -1]].tolist()
        # Remember how many leading columns were trimmed, so callers can place the
        # trimmed lines back at their true x position (e.g. overlay plotting).
        self.x_offset = first
        return lines[:, first : last + 1]

    def extract_endpoints(self, lines: torch.Tensor) -> tuple[list[int], list[int], list[float], list[float]]:
        xmin: list[int] = []
        xmax: list[int] = []
        ymin: list[float] = []
        ymax: list[float] = []
        for line in lines:
            if torch.all(torch.isnan(line)):
                continue
            valid = ~torch.isnan(line)
            start, end = valid.nonzero()[0], valid.nonzero()[-1]
            xmin.append(int(start.item()))
            xmax.append(int(end.item()))
            ymin.append(float(line[start].item()))
            ymax.append(float(line[end].item()))
        return xmin, xmax, ymin, ymax

    def extract_graph_params(self, lines: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        W: int = lines.shape[1]
        xmin, xmax, ymin, ymax = self.extract_endpoints(lines)
        min_coords = torch.tensor(np.column_stack((xmin, ymin)), dtype=torch.float32)
        max_coords = torch.tensor(np.column_stack((xmax, ymax)), dtype=torch.float32)
        heights = torch.nanmean(lines, dim=1).abs()
        return min_coords, max_coords, heights, W

    def compute_cost_matrix(
        self, min_coords: torch.Tensor, max_coords: torch.Tensor, W: int, heights: torch.Tensor
    ) -> tuple[npt.NDArray[Any], torch.Tensor]:
        lam = self.lam
        N = min_coords.shape[0]
        min_exp = min_coords.unsqueeze(1).expand(N, N, 2)
        max_exp = max_coords.unsqueeze(0).expand(N, N, 2)

        delta_x = min_exp[..., 0] - max_exp[..., 0]
        wrapped_x = torch.minimum(delta_x.abs(), W - delta_x.abs()) * lam
        wrapped_mask = (W - delta_x.abs()) < delta_x.abs()

        delta_x = torch.where(delta_x < 0, delta_x / lam, delta_x)
        wrapped_x = torch.where(wrapped_mask, wrapped_x, delta_x.abs())

        delta_y = min_exp[..., 1] - max_exp[..., 1]
        delta_y = torch.where(wrapped_mask, delta_y * lam, delta_y)

        distances = wrapped_x.abs() + delta_y.abs()

        heights_norm = (heights - heights.min()) / heights.max()
        heights_diff = torch.abs(heights_norm.unsqueeze(1) - heights_norm.unsqueeze(0))

        cost_matrix: npt.NDArray[Any] = (
            distances * (1 + heights_diff * self.height_difference_penalty)
        ).numpy()
        return cost_matrix, wrapped_mask

    def match_lines(self, cost_matrix: npt.NDArray[Any]) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return row_ind, col_ind

    def build_match_graph(self, row_ind: npt.NDArray[Any], col_ind: npt.NDArray[Any]) -> dict[int, list[int]]:
        graph: dict[int, list[int]] = defaultdict(list)
        row_ind = np.atleast_1d(row_ind)
        col_ind = np.atleast_1d(col_ind)
        for i, j in zip(row_ind, col_ind):
            graph[i].append(j)
            graph[j].append(i)
        return graph

    def get_connected_components(self, graph: dict[int, list[int]]) -> list[list[int]]:
        visited: set[int] = set()
        components: list[list[int]] = []
        for node in graph:
            if node in visited:
                continue
            queue = deque([node])
            comp: list[int] = []
            while queue:
                curr = queue.popleft()
                if curr in visited:
                    continue
                visited.add(curr)
                comp.append(curr)
                for n in graph[curr]:
                    if n not in visited:
                        queue.append(n)
            components.append(comp)
        return components

    def merge_components(
        self, lines: torch.Tensor, components: list[list[int]]
    ) -> tuple[list[torch.Tensor], list[float]]:
        merged: list[torch.Tensor] = []
        overlaps: list[float] = []
        for group in components:
            group_lines = lines[torch.tensor(group)]
            valid_mask = ~torch.isnan(group_lines)
            merged_line = torch.full((group_lines.shape[1],), float("nan"))
            overlap = valid_mask.sum(0)
            overlaps.append(float(overlap[overlap > 0].float().mean().item()))
            for col in range(group_lines.shape[1]):
                valid_values = group_lines[:, col][~torch.isnan(group_lines[:, col])]
                if len(valid_values) > 0:
                    merged_line[col] = valid_values[0]
            merged.append(merged_line)
        return merged, overlaps

    def plot_graph(self, min_coords: torch.Tensor, max_coords: torch.Tensor, row_ind: Any, col_ind: Any) -> None:
        plt.figure(figsize=(10, 6))
        for i, j in zip(row_ind, col_ind):
            plt.plot([min_coords[i, 0], max_coords[j, 0]], [min_coords[i, 1], max_coords[j, 1]], color="red")
        plt.scatter(min_coords[:, 0], min_coords[:, 1], color="blue", label="Min Coords")
        plt.scatter(max_coords[:, 0], max_coords[:, 1], color="green", label="Max Coords")
        plt.title("Graph of Merged Lines")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.gca().invert_yaxis()
        plt.legend()
        plt.savefig("sandbox/graph_of_merged_lines.png")
        plt.close()

    def plot_lines(self, lines: torch.Tensor, title: str) -> None:
        plt.figure(figsize=(10, 6))
        for line in lines:
            if torch.all(torch.isnan(line)):
                continue
            plt.plot(line.numpy(), alpha=0.7, linewidth=1.5)
        plt.title(title)
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.gca().invert_yaxis()
        plt.savefig(f"sandbox/{title.replace(' ', '_').casefold()}.png")
        plt.close()

    def match_and_merge_lines(self, lines: torch.Tensor) -> tuple[list[torch.Tensor], list[float]]:
        lines = self.preprocess_lines(lines)
        if self.debug:
            self.plot_lines(lines, "Preprocessed Lines")
        min_coords, max_coords, heights, W = self.extract_graph_params(lines)
        cost_matrix, wrapped_mask = self.compute_cost_matrix(min_coords, max_coords, W, heights)

        row_ind, col_ind = self.match_lines(cost_matrix)

        valid_mask = ~wrapped_mask[row_ind, col_ind]
        row_ind, col_ind = row_ind[valid_mask], col_ind[valid_mask]

        graph = self.build_match_graph(row_ind, col_ind)
        components = self.get_connected_components(graph)

        merged_lines, overlaps = self.merge_components(lines, components)
        filtered_lines = [line for line in merged_lines if torch.sum(~torch.isnan(line)) >= W // 5]

        if self.debug:
            self.plot_graph(min_coords, max_coords, row_ind, col_ind)

        return filtered_lines, overlaps
