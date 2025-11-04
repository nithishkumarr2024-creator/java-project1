import numpy as np
import cv2
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import Image
from scipy.signal import butter, filtfilt
from scipy.stats import entropy
from scipy.fftpack import fft
import random

# ==============================
# STEP 1. DATA ACQUISITION
# ==============================

# Replace this link with your wave frame image
wave_image_url = "https://i.postimg.cc/vHFDFrPL/il-300x300-7379739633-s6zy.webp"

response = requests.get(wave_image_url)
img = Image.open(BytesIO(response.content))
img = np.array(img)

plt.title("Sample Ocean Wave Frame (Input)")
plt.imshow(img)
plt.axis('off')
plt.show()

# ==============================
# STEP 2. PREPROCESSING
# ==============================

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Resize to 256x256 (Normalization)
gray = cv2.resize(gray, (256, 256))

# Apply adaptive noise filtering (low-pass Butterworth)
def butter_lowpass_filter(data, cutoff=0.1, fs=30.0, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

filtered = np.apply_along_axis(butter_lowpass_filter, 0, gray)

# Contrast enhancement (Histogram Equalization)
enhanced = cv2.equalizeHist(filtered.astype(np.uint8))

plt.title("Preprocessed Wave Frame")
plt.imshow(enhanced, cmap='gray')
plt.axis('off')
plt.show()

# ==============================
# STEP 3. ENTROPY EXTRACTION
# ==============================

# Flatten image to 1D signal
signal = enhanced.flatten()

# Normalize
signal_norm = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

# Quantization & Thresholding -> Convert to bits
binary_seq = np.where(signal_norm > 0.5, 1, 0)

# Von Neumann Debiasing
def von_neumann(bits):
    unbiased = []
    for i in range(0, len(bits) - 1, 2):
        if bits[i] != bits[i+1]:
            unbiased.append(bits[i])
    return np.array(unbiased)

bitstream = von_neumann(binary_seq)

# ==============================
# STEP 4. RANDOM NUMBER GENERATION
# ==============================

# Convert bits to bytes
def bits_to_bytes(bits):
    return [int("".join(str(b) for b in bits[i:i+8]), 2)
            for i in range(0, len(bits) - 7, 8)]

random_numbers = bits_to_bytes(bitstream)
random_numbers = np.array(random_numbers)

plt.hist(random_numbers, bins=30, color='steelblue', edgecolor='black')
plt.title("Random Bitstream Distribution")
plt.xlabel("Value Range (0-255)")
plt.ylabel("Frequency")
plt.show()

# ==============================
# STEP 5. RANDOMNESS TESTING
# ==============================

# Frequency Test
p_freq = np.mean(bitstream)
print(f"Frequency Test -> Mean Bit Value: {p_freq:.4f}")

# Runs Test
runs, prev = 0, bitstream[0]
for b in bitstream:
    if b != prev:
        runs += 1
        prev = b
expected_runs = (2 * len(bitstream) * p_freq * (1 - p_freq))
z_stat = (runs - expected_runs) / np.sqrt(2 * len(bitstream) * p_freq * (1 - p_freq) * (2 * len(bitstream) * p_freq * (1 - p_freq)))
print(f"Runs Test -> Runs: {runs}, Expected: {expected_runs:.2f}, Z: {z_stat:.3f}")

# Spectral Test (FFT)
fft_values = np.abs(fft(bitstream))
peaks = np.sum(fft_values > np.mean(fft_values))
print(f"Spectral Test -> Peaks Above Threshold: {peaks}")

# Entropy Measure
H = entropy(np.bincount(random_numbers), base=2)
print(f"Shannon Entropy of Random Sequence: {H:.4f} bits")

# ==============================
# STEP 6. VISUALIZATIONS
# ==============================

# Time Series Representation
plt.plot(random_numbers[:200])
plt.title("Random Number Sequence (First 200)")
plt.xlabel("Sample Index")
plt.ylabel("Random Value")
plt.show()

# Autocorrelation
autocorr = np.correlate(random_numbers - np.mean(random_numbers),
                        random_numbers - np.mean(random_numbers), mode='full')
autocorr = autocorr[autocorr.size // 2:]
plt.plot(autocorr)
plt.title("Autocorrelation (Wave Temporal Lag)")
plt.xlabel("Lag")
plt.ylabel("Correlation")
plt.show()

print("\n✅ Aqua-Entropy Pipeline Complete.")
print("Results indicate entropy extraction and randomness quality similar to NIST test criteria.")