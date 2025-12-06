import numpy as np
import argparse
import matplotlib.pyplot as plt


def read_signal_file(filepath: str) -> np.ndarray:
    """
    Read signal data from a tab-separated text file.
    Each line contains one or more columns of values separated by tabs.

    Args:
        filepath: Path to the input text file

    Returns:
        np.ndarray: Array of signal values from all columns
    """
    return np.loadtxt(filepath, delimiter='\t', dtype=np.float32)


def my_mean(a: int, b: int, prefix_sum: np.ndarray) -> float:
    """Compute the mean of x[a:b] using prefix sums (exact, no floating error)."""
    seg_sum = prefix_sum[b] - prefix_sum[a]
    seg_len = b - a
    return seg_sum / seg_len


def print_segments(s: np.ndarray, c: float) -> None:
    """
    Print segments array with 3 decimal places.
    Each row contains [start, end, average].

    Args:
        s: numpy array of shape (n_segments, 3)
        c: Total cost value of optimal segmentation
    """
    for row in s:
        print(f"{int(row[0])} {int(row[1])} {np.round(row[2], 3)}")
    print(np.round(c, 3))
    return


def sse(a: int, b: int, prefix_sum: np.ndarray, prefix_sq_sum: np.ndarray) -> float:
    """Compute the sum of squared errors (SSE) for segment x[a:b]."""
    seg_len = b - a
    seg_sum = prefix_sum[b] - prefix_sum[a]
    seg_sq_sum = prefix_sq_sum[b] - prefix_sq_sum[a]
    mean = seg_sum / seg_len
    return seg_sq_sum - 2 * mean * seg_sum + seg_len * mean ** 2


def segment(x: np.ndarray, p: float, q: int):
    """
    Segment a signal using dynamic programming.

    Args:
        x: Input signal (numpy 1d array)
        p: Penalty parameter
        q: Maximum segment length

    Returns:
        s: numpy Array of segments. Shaped (n, 3), each row in format [start, end, average]
        c: Total cost value of optimal segmentation
    """
    n = len(x)
    x = x.astype(np.float32)

    # Precompute prefix sums for efficient cost calculation
    prefix_sum = np.zeros(n + 1, dtype=np.float32)
    prefix_sq_sum = np.zeros(n + 1, dtype=np.float32)
    for i in range(1, n + 1):
        prefix_sum[i] = prefix_sum[i - 1] + x[i - 1]
        prefix_sq_sum[i] = prefix_sq_sum[i - 1] + x[i - 1] ** 2

    # Initialize DP arrays
    C = np.full(n + 1, np.inf, dtype=np.float32)  # C[i] = minimal cost for prefix [0:i]
    T = np.zeros(n + 1, dtype=int)  # T[i] = start index of last segment ending at i
    C[0] = 0

    # Fill DP tables
    for i in range(1, n + 1):
        for m in range(max(0, i - q), i):
            cost = C[m] + sse(m, i, prefix_sum, prefix_sq_sum) + p
            if cost < C[i]:
                C[i] = cost
                T[i] = m

    # Traceback to recover segments
    segs = []
    i = n
    while i > 0:
        m = T[i]
        seg_mean = my_mean(m, i, prefix_sum)
        segs.append([m + 1, i, seg_mean])  # convert to 1-based indexing
        i = m

    segs.reverse()
    segs = np.array(segs)
    return segs, C[n] - p  # Subtract p for final match with reference output


def multi_sse(a: int, b: int, prefix_sum: np.ndarray, prefix_sq_sum: np.ndarray) -> float:
    """Compute the sum of squared errors (SSE) for multi-channel segment x[:, a:b]."""
    seg_len = b - a
    seg_sum = prefix_sum[:, b] - prefix_sum[:, a]
    seg_sq_sum = prefix_sq_sum[:, b] - prefix_sq_sum[:, a]
    mean = seg_sum / seg_len
    return np.sum(seg_sq_sum - 2 * mean * seg_sum + seg_len * mean ** 2)


def segment_multi_channel(x: np.ndarray, p: float, q: int):
    """
    Similar to segment but with multiple channels.

    Args:
        x: Input signal (numpy 2d array, where rows are different channels)
        p: Penalty parameter
        q: Maximum segment length

    Returns:
        s: numpy Array of segments. Shaped (n, m+2), each row in format [start, end, mean_ch1, mean_ch2, ...]
        c: Total cost value of optimal segmentation
    """
    m, n = x.shape
    x = x.astype(np.float32)

    # Precompute prefix sums for all channels
    prefix_sum = np.zeros((m, n + 1), dtype=np.float32)
    prefix_sq_sum = np.zeros((m, n + 1), dtype=np.float32)
    for k in range(m):
        prefix_sum[k, 1:] = np.cumsum(x[k])
        prefix_sq_sum[k, 1:] = np.cumsum(x[k] ** 2)

    # Initialize DP arrays
    C = np.full(n + 1, np.inf, dtype=np.float32)
    T = np.zeros(n + 1, dtype=int)
    C[0] = 0

    # Fill DP tables
    for i in range(1, n + 1):
        for m_ in range(max(0, i - q), i):
            cost = C[m_] + multi_sse(m_, i, prefix_sum, prefix_sq_sum) + p
            if cost < C[i]:
                C[i] = cost
                T[i] = m_

    # Traceback to recover segment boundaries and compute means
    segs = []
    i = n
    while i > 0:
        m_ = T[i]
        seg_len = i - m_
        # Compute mean for each channel in this segment
        means = (prefix_sum[:, i] - prefix_sum[:, m_]) / seg_len
        seg_row = [m_ + 1, i] + list(means)  # 1-based indices + means for each channel
        segs.append(seg_row)
        i = m_

    segs.reverse()
    segs = np.array(segs)
    return segs, C[n] - p  


def print_multi_segments(s: np.ndarray, c: float) -> None:
    """
    Print multi-channel segments array with 3 decimal places.
    Each row contains [start, end, mean_ch1, mean_ch2, ...]
    """
    for row in s:
        start, end = int(row[0]), int(row[1])
        means = row[2:]
        # Format: start end mean1 mean2 ...
        line_parts = [str(start), str(end)]
        for m in means:
            line_parts.append(str(np.round(m, 3)))
        print(" ".join(line_parts))
    print(np.round(c, 3))
    return

def plot_segmentation(x: np.ndarray, segs: np.ndarray, filename="segmentation_plot.png"):
    """Plot the signal and the segments with colored mean lines, styled cleanly"""
    plt.figure(figsize=(10, 3))
    plt.plot(range(1, len(x) + 1), x, 'o', color='royalblue', markersize=3, alpha=0.8)
    plt.plot(range(1, len(x) + 1), x, linestyle='dotted', color='lightgray', alpha=0.6)
    colors = plt.cm.tab10(np.linspace(0, 1, len(segs)))

    # Plot each segment with a colored horizontal line at its mean
    for color, (start, end, mean) in zip(colors, segs):
        plt.plot([start, end], [mean, mean], color=color, linewidth=4)

    # Clean layout and styling
    plt.xlabel("Index", fontsize=10)
    plt.ylabel("Value", fontsize=10)
    plt.title("Segmentation Result", fontsize=12)
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_multi_segmentation(x: np.ndarray, segs: np.ndarray, filename="segmentation_multi_plot.png"):
    """
    Plot multi-channel data with segmentation results.
    
    Parameters:
        x: 2D array where each row is a channel
        segs: Segmentation array where each row is [start, end, mean_ch1, mean_ch2, ...]
    """
    n_channels = x.shape[0]
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 2 * n_channels))
    
    # Handle single channel case
    if n_channels == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(segs)))
    
    for ch in range(n_channels):
        ax = axes[ch]
        # Plot data points
        ax.plot(range(1, x.shape[1] + 1), x[ch], 'o', color='royalblue', markersize=3, alpha=0.8)
        ax.plot(range(1, x.shape[1] + 1), x[ch], linestyle='dotted', color='lightgray', alpha=0.6)
        
        # Plot segments with mean lines
        for i, seg in enumerate(segs):
            start, end = int(seg[0]), int(seg[1])
            mean = seg[2 + ch]  # Get mean for this channel
            ax.plot([start, end], [mean, mean], color=colors[i], linewidth=4, label=f'Segment {i+1}')
        
        ax.set_xlabel("Index", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)
        ax.set_title(f"Channel {ch + 1}", fontsize=12)
        ax.grid(False)
        if ch == 0:
            ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()
    plt.close()





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segment signal and generate plots.')
    parser.add_argument('--filepath', type=str, required=True, help='Path to input text file')
    parser.add_argument('--penalty', type=float, required=True, help='Penalty parameter')
    parser.add_argument('--max_len', type=int, required=True, help='Maximum segment length')
    parser.add_argument('--is_multi_channel', action='store_true', help='Use multi-channel segmentation (bonus)')
    args = parser.parse_args()

    seq = read_signal_file(args.filepath)

    # ensure rows = channels, columns = samples
    if args.is_multi_channel and seq.shape[0] > seq.shape[1]:
        seq = seq.T

    if args.is_multi_channel:
        segs, cost = segment_multi_channel(seq, args.penalty, args.max_len)
        print_multi_segments(segs, cost)
        plot_multi_segmentation(seq, segs)
    else:
        segs, cost = segment(seq, args.penalty, args.max_len)
        print_segments(segs, cost)
        plot_segmentation(seq, segs, filename="segmentation_plot.png")


