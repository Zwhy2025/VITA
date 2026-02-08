import numpy as np

def compute_stats(data: np.ndarray, is_image: bool = False) -> dict[str, np.ndarray]:
    if not isinstance(data, np.ndarray):
        raise ValueError(f"Data must be a numpy array, but got {type(data)} instead.")
    if data.ndim == 0:
        raise ValueError("Data must have at least 1 dimension.")
    
    count = np.array([data.shape[0]], dtype=np.float64)
    
    if is_image:
        if data.ndim == 4 and data.shape[-1] == 3:
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
    means = np.stack([s["mean"] for s in stats_ft_list])
    variances = np.stack([s["std"] ** 2 for s in stats_ft_list])
    counts = np.stack([s["count"] for s in stats_ft_list])
    total_count = counts.sum(axis=0)

    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, axis=-1)

    weighted_means = means * counts
    total_mean = weighted_means.sum(axis=0) / total_count

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
    _assert_type_and_shape(stats_list)

    data_keys = {key for stats in stats_list for key in stats}
    aggregated_stats = {key: {} for key in data_keys}

    for key in data_keys:
        stats_with_key = [stats[key] for stats in stats_list if key in stats]
        aggregated_stats[key] = aggregate_feature_stats(stats_with_key)

    return aggregated_stats
