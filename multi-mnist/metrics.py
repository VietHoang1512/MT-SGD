import numpy as np
import seaborn as sns

sns.set()


def brier_score(y_true, y_pred):
    return 1 + (np.sum(y_pred ** 2) - 2 * np.sum(y_pred[np.arange(y_pred.shape[0]), y_true])) / y_true.shape[0]


def compute_calibration(true_labels, pred_labels, confidences, num_bins=20):

    assert len(confidences) == len(pred_labels)
    assert len(confidences) == len(true_labels)
    assert num_bins > 0

    # bin_size = 1.0 / num_bins
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(confidences, bins, right=True)

    bin_accuracies = np.zeros(num_bins, dtype=np.float)
    bin_confidences = np.zeros(num_bins, dtype=np.float)
    bin_counts = np.zeros(num_bins, dtype=np.int)

    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            bin_counts[b] = len(selected)

    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    # mce = np.max(gaps)

    return ece
