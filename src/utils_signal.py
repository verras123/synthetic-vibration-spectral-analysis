# src/utils_signal.py
import numpy as np
from scipy.signal import welch, find_peaks

def unilateral_fft(sig, fs_hz):
    """
    Unilateral FFT amplitude spectrum using Hann window.
    Returns (freqs_hz, amp).
    """
    sig = np.asarray(sig, dtype=float)
    n = sig.size
    if n < 2:
        return np.array([0.0]), np.array([0.0])

    sig = sig - np.mean(sig)
    w = np.hanning(n)
    xw = sig * w

    x = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(fs_hz))

    amp = np.abs(x) / n
    # One-sided correction
    if n % 2 == 0:
        if amp.size > 2:
            amp[1:-1] *= 2.0
    else:
        if amp.size > 1:
            amp[1:] *= 2.0

    return freqs, amp

def amplitude_spectral_density(sig, fs_hz, nperseg=256, noverlap=None):
    """
    ASD via Welch: returns (freqs_hz, asd).
    """
    sig = np.asarray(sig, dtype=float)
    n = sig.size
    if n < 2:
        return np.array([0.0]), np.array([0.0])

    sig = sig - np.mean(sig)

    nperseg_adj = int(min(max(8, nperseg), n))
    if noverlap is None:
        noverlap_adj = nperseg_adj // 2
    else:
        noverlap_adj = int(min(max(0, noverlap), nperseg_adj - 1))

    freqs, psd = welch(
        sig,
        fs=float(fs_hz),
        window="hann",
        nperseg=nperseg_adj,
        noverlap=noverlap_adj,
        scaling="density"
    )
    asd = np.sqrt(psd)
    return freqs, asd

def detect_peaks(y, prominence_ratio=0.05):
    """
    Peak detection using prominence as fraction of max(y).
    Returns indices array.
    """
    y = np.asarray(y, dtype=float)
    if y.size < 3:
        return np.array([], dtype=int)

    y_max = float(np.max(y))
    if y_max <= 0.0:
        return np.array([], dtype=int)

    prom = float(prominence_ratio) * y_max
    idx, _ = find_peaks(y, prominence=prom)
    return idx
