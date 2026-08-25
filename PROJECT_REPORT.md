# SP25 Group Project Report — Line Code Analyzer

## 1. Objective
Develop a web application that generates and analyzes digital communication line-coded waveforms.

## 2. Line Codes
NRZ, RZ and Manchester are supported as alternative line-coding choices.

## 3. Pulse Shaping
Rectangular, Gaussian and Root Raised Cosine (RRC) shaping are supported as alternative choices.

## 4. Channel
Additive White Gaussian Noise (AWGN) models a noisy communication channel. SNR controls the relative noise level.

## 5. Analysis
The received signal is analyzed in time and frequency domains, and its energy, average power, RMS, peak and eye diagram are displayed.

## 6. Expected Observations
Lower SNR produces a noisier waveform and generally a less open eye. Smoother pulse shaping changes abrupt transitions and high-frequency spectral content. Manchester has a transition in every bit and therefore has different spectral behavior from NRZ.

## 7. Technology
Python, NumPy, SciPy, Matplotlib, Pandas and Streamlit.

## 8. Method
Bits → line code → pulse shaping → AWGN → received signal → spectrum/metrics/eye diagram.

## 9. Future Scope
BER calculation, matched filtering, equalization, more line codes, modulation schemes and mobile frontend.
