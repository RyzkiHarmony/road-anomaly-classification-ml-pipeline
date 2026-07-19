import numpy as np

def get_stratified_group_split(groups, y_raw, train_ratio=0.7):
    groups = np.asarray(groups)
    y_raw = np.asarray(y_raw)

    if groups.shape[0] != y_raw.shape[0]:
        raise ValueError("groups and y_raw must have the same length")
    if groups.shape[0] == 0:
        raise ValueError("groups and y_raw must not be empty")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")

    unique_classes = np.unique(y_raw)
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    y_idx = np.array([class_to_idx[val] for val in y_raw])

    group_names = np.unique(groups)
    group_counts = {g: np.array([np.sum(y_idx[groups == g] == i) for i in range(len(unique_classes))]) for g in group_names}
    total_counts = np.sum(list(group_counts.values()), axis=0)

    train_groups = set()
    test_groups = set()
    current_train = np.zeros(len(unique_classes))

    # Sort groups by total minority class count desc
    minority_indices = [class_to_idx[c] for c in ['Pothole', 'Speed Bump'] if c in class_to_idx]
    sorted_groups = sorted(group_names, key=lambda g: np.sum(group_counts[g][minority_indices]), reverse=True)

    for g in sorted_groups:
        counts = group_counts[g]
        # Minimize MSE of ratios to train_ratio:
        # If added to train:
        ratio_if_train = (current_train + counts) / (total_counts + 1e-9)
        err_train = np.sum((ratio_if_train - train_ratio) ** 2)
        
        # If added to test:
        ratio_if_test = current_train / (total_counts + 1e-9)
        err_test = np.sum((ratio_if_test - train_ratio) ** 2)
        
        if err_train < err_test:
            train_groups.add(g)
            current_train += counts
        else:
            test_groups.add(g)

    if not train_groups or not test_groups:
        raise ValueError("Split failed: train_groups or test_groups is empty")

    return sorted(train_groups), sorted(test_groups)
