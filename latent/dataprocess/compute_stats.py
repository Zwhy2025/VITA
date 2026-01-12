"""
从 lerobot/lerobot/common/datasets/compute_stats.py 复制
"""
import numpy as np


def compute_stats(data: np.ndarray, is_image: bool = False) -> dict[str, np.ndarray]:
    """Compute statistics (min, max, mean, std, count) from a numpy array.
    
    Args:
        data: Input numpy array of shape (N, ...) where N is the number of samples.
        is_image: If True, treat data as image and compute per-channel stats with shape (3, 1, 1).
                  If False, compute stats along the first dimension preserving other dimensions.
        
    Returns:
        Dictionary containing:
            - "min": Minimum values
            - "max": Maximum values  
            - "mean": Mean values
            - "std": Standard deviation
            - "count": Total count (shape: (1,))
    """
    if not isinstance(data, np.ndarray):
        raise ValueError(f"Data must be a numpy array, but got {type(data)} instead.")
    if data.ndim == 0:
        raise ValueError("Data must have at least 1 dimension.")
    
    count = np.array([data.shape[0]], dtype=np.float64)
    
    if is_image:
        # For image data: compute stats per channel and reshape to (3, 1, 1)
        if data.ndim == 4 and data.shape[-1] == 3:
            # (N, H, W, 3) -> compute stats for each channel
            data_reshaped = data.reshape(-1, 3)
            min_vals = np.min(data_reshaped, axis=0)
            max_vals = np.max(data_reshaped, axis=0)
            mean_vals = np.mean(data_reshaped, axis=0)
            std_vals = np.std(data_reshaped, axis=0)
            return {
                "min": min_vals.reshape(3, 1, 1),
                "max": max_vals.reshape(3, 1, 1),
                "mean": mean_vals.reshape(3, 1, 1),
                "std": std_vals.reshape(3, 1, 1),
                "count": count,
            }
        else:
            raise ValueError(f"For image data, expected shape (N, H, W, 3), but got {data.shape}")
    
    # For non-image data, compute stats along the first dimension
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    mean_vals = np.mean(data, axis=0)
    std_vals = np.std(data, axis=0)
    
    return {
        "min": min_vals,
        "max": max_vals,
        "mean": mean_vals,
        "std": std_vals,
        "count": count,
    }


def _assert_type_and_shape(stats_list: list[dict[str, dict]]):
    for i in range(len(stats_list)):
        for fkey in stats_list[i]:
            for k, v in stats_list[i][fkey].items():
                if not isinstance(v, np.ndarray):
                    raise ValueError(
                        f"Stats must be composed of numpy array, but key '{k}' of feature '{fkey}' is of type '{type(v)}' instead."
                    )
                if v.ndim == 0:
                    raise ValueError("Number of dimensions must be at least 1, and is 0 instead.")
                if k == "count" and v.shape != (1,):
                    raise ValueError(f"Shape of 'count' must be (1), but is {v.shape} instead.")
                if "image" in fkey and k != "count" and v.shape != (3, 1, 1):
                    raise ValueError(f"Shape of '{k}' must be (3,1,1), but is {v.shape} instead.")


def aggregate_feature_stats(stats_ft_list: list[dict[str, dict]]) -> dict[str, dict[str, np.ndarray]]:
    """Aggregates stats for a single feature."""
    means = np.stack([s["mean"] for s in stats_ft_list])
    variances = np.stack([s["std"] ** 2 for s in stats_ft_list])
    counts = np.stack([s["count"] for s in stats_ft_list])
    total_count = counts.sum(axis=0)

    # Prepare weighted mean by matching number of dimensions
    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, axis=-1)

    # Compute the weighted mean
    weighted_means = means * counts
    total_mean = weighted_means.sum(axis=0) / total_count

    # Compute the variance using the parallel algorithm
    delta_means = means - total_mean
    weighted_variances = (variances + delta_means**2) * counts
    total_variance = weighted_variances.sum(axis=0) / total_count

    return {
        "min": np.min(np.stack([s["min"] for s in stats_ft_list]), axis=0),
        "max": np.max(np.stack([s["max"] for s in stats_ft_list]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "count": total_count,
    }


def aggregate_stats(stats_list: list[dict[str, dict]]) -> dict[str, dict[str, np.ndarray]]:
    """Aggregate stats from multiple compute_stats outputs into a single set of stats.

    The final stats will have the union of all data keys from each of the stats dicts.

    For instance:
    - new_min = min(min_dataset_0, min_dataset_1, ...)
    - new_max = max(max_dataset_0, max_dataset_1, ...)
    - new_mean = (mean of all data, weighted by counts)
    - new_std = (std of all data)
    """

    _assert_type_and_shape(stats_list)

    data_keys = {key for stats in stats_list for key in stats}
    aggregated_stats = {key: {} for key in data_keys}

    for key in data_keys:
        stats_with_key = [stats[key] for stats in stats_list if key in stats]
        aggregated_stats[key] = aggregate_feature_stats(stats_with_key)

    return aggregated_stats
