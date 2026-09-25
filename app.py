import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Line Code Analyzer", page_icon="📡", layout="wide")

st.title("📡 Line Code Analyzer")


def check_bits(bits):
    bits = bits.replace(" ", "").strip()

    if bits == "":
        return None

    for x in bits:
        if x not in "01":
            return None

    return bits


def nrz(bits, sps):
    signal = []

    for bit in bits:
        value = 1 if bit == "1" else -1

        for _ in range(sps):
            signal.append(value)

    return np.array(signal)


def rz(bits, sps):
    signal = []
    half = sps // 2

    for bit in bits:
        value = 1 if bit == "1" else -1

        for _ in range(half):
            signal.append(value)

        for _ in range(sps - half):
            signal.append(0)

    return np.array(signal)


def manchester(bits, sps):
    signal = []
    half = sps // 2

    for bit in bits:
        if bit == "1":
            first = 1
            second = -1
        else:
            first = -1
            second = 1

        for _ in range(half):
            signal.append(first)

        for _ in range(sps - half):
            signal.append(second)

    return np.array(signal)


def gaussian(signal, sps):
    size = max(5, sps // 2)

    if size % 2 == 0:
        size += 1

    x = np.linspace(-2, 2, size)
    sigma = 0.7

    pulse = np.exp(-(x ** 2) / (2 * sigma ** 2))
    pulse = pulse / np.sum(pulse)

    result = np.convolve(signal, pulse, mode="same")

    m = np.max(np.abs(result))

    if m != 0:
        result = result / m

    return result


def rrc(sps, beta=0.35, span=6):
    t = np.arange(
        -span * sps / 2,
        span * sps / 2 + 1
    ) / sps

    h = np.zeros(len(t))

    for i in range(len(t)):
        x = t[i]

        if abs(x) < 1e-10:
            h[i] = 1 + beta * (4 / np.pi - 1)

        elif (
            beta != 0
            and abs(abs(x) - 1 / (4 * beta)) < 1e-10
        ):
            h[i] = (
                beta / np.sqrt(2)
                * (
                    (1 + 2 / np.pi)
                    * np.sin(np.pi / (4 * beta))
                    +
                    (1 - 2 / np.pi)
                    * np.cos(np.pi / (4 * beta))
                )
            )

        else:
            a = np.sin(np.pi * x * (1 - beta))
            b = 4 * beta * x * np.cos(np.pi * x * (1 + beta))
            c = np.pi * x * (1 - (4 * beta * x) ** 2)

            h[i] = (a + b) / c

    h = h / np.sqrt(np.sum(h ** 2))

    return h


def pulse_shape(signal, option, sps):

    if option == "Rectangular":
        return signal.copy()

    if option == "Gaussian":
        return gaussian(signal, sps)

    if option == "RRC":
        filter_value = rrc(sps)

        result = np.convolve(
            signal,
            filter_value,
            mode="same"
        )

        m = np.max(np.abs(result))

        if m != 0:
            result = result / m

        return result


def add_noise(signal, snr_db, seed):
    rng = np.random.default_rng(seed)

    power = np.mean(signal ** 2)

    snr_linear = 10 ** (snr_db / 10)

    noise_power = power / snr_linear

    noise = rng.normal(
        0,
        np.sqrt(noise_power),
        len(signal)
    )

    return signal + noise


def fft_signal(signal):
    n = len(signal)

    y = np.fft.fft(signal)
    f = np.fft.fftfreq(n)

    magnitude = np.abs(y) / n

    index = f >= 0

    return f[index], magnitude[index]


def signal_values(signal):
    energy = np.sum(signal ** 2)
    power = np.mean(signal ** 2)
    rms = np.sqrt(power)
    peak = np.max(np.abs(signal))

    return energy, power, rms, peak


def eye_data(signal, sps):
    length = 2 * sps
    traces = []

    count = min(100, len(signal) // sps - 1)

    for i in range(count):
        start = i * sps
        part = signal[start:start + length]

        if len(part) == length:
            traces.append(part)

    if len(traces) == 0:
        return None

    return np.array(traces)


st.sidebar.header("Settings")

sps = st.sidebar.slider(
    "Samples per bit",
    10,
    100,
    50,
    10
)

coding = st.sidebar.selectbox(
    "Line Coding",
    ["NRZ", "RZ", "Manchester"]
)

pulse = st.sidebar.selectbox(
    "Pulse Shaping",
    ["Rectangular", "Gaussian", "RRC"]
)

snr = st.sidebar.slider(
    "SNR (dB)",
    0,
    30,
    20
)

seed = st.sidebar.number_input(
    "Noise Seed",
    0,
    10000,
    42
)


st.subheader("Binary Input")

bits_input = st.text_input(
    "Enter binary data",
    "10110010"
)

st.caption("Use only 0 and 1. Spaces are allowed.")


if st.button("Generate Signal"):

    bits = check_bits(bits_input)

    if bits is None:
        st.error("Invalid input. Enter only 0 and 1.")
        st.stop()

    if len(bits) > 2000:
        st.error("Please enter 2000 bits or fewer.")
        st.stop()

    if coding == "NRZ":
        coded = nrz(bits, sps)

    elif coding == "RZ":
        coded = rz(bits, sps)

    else:
        coded = manchester(bits, sps)

    shaped = pulse_shape(
        coded,
        pulse,
        sps
    )

    received = add_noise(
        shaped,
        snr,
        seed
    )

    time = np.arange(len(coded)) / sps

    st.success("Signal generated successfully.")

    st.write("Input:", bits)
    st.write("Number of bits:", len(bits))

    st.subheader("Line-Coded Signal")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time, coded)
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.set_title(coding)
    ax.grid()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Pulse-Shaped Signal")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time, shaped)
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.set_title(pulse + " Pulse Shaping")
    ax.grid()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Received Signal")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time, received)
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.set_title("Received Signal")
    ax.grid()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Frequency Spectrum")

    frequency, magnitude = fft_signal(received)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(frequency, magnitude)
    ax.set_xlabel("Normalized Frequency")
    ax.set_ylabel("Magnitude")
    ax.set_title("FFT Spectrum")
    ax.grid()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Signal Measurements")

    energy, power, rms, peak = signal_values(received)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Energy", f"{energy:.3f}")
    c2.metric("Average Power", f"{power:.3f}")
    c3.metric("RMS", f"{rms:.3f}")
    c4.metric("Peak", f"{peak:.3f}")

    st.subheader("Eye Diagram")

    eye = eye_data(received, sps)

    if eye is not None:

        fig, ax = plt.subplots(figsize=(10, 5))

        x = np.arange(2 * sps) / sps

        for row in eye:
            ax.plot(x, row, alpha=0.35)

        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        ax.set_title("Eye Diagram")
        ax.grid()

        st.pyplot(fig)
        plt.close(fig)

    else:
        st.warning("Not enough data for eye diagram.")

    st.subheader("Download Data")

    data = pd.DataFrame({
        "Time": time,
        "Line_Code": coded,
        "Pulse_Shaped": shaped,
        "Received": received
    })

    csv = data.to_csv(index=False)

    st.download_button(
        "Download CSV",
        csv,
        "line_code_results.csv",
        "text/csv"
    )
