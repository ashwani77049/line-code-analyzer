import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Line Code Analyzer",
    page_icon="📡",
    layout="wide"
)

st.title("📡 Line Code Analyzer")
st.caption("SP25 Digital Communication Group Project")


# =====================================================
# LINE CODING
# =====================================================

def line_code(bits, code, sps):

    signal = []

    for bit in bits:

        if code == "NRZ":
            level = 1 if bit == "1" else -1
            signal += [level] * sps

        elif code == "RZ":
            level = 1 if bit == "1" else -1
            half = sps // 2
            signal += [level] * half
            signal += [0] * (sps - half)

        elif code == "Manchester":
            half = sps // 2

            if bit == "1":
                signal += [1] * half
                signal += [-1] * (sps - half)
            else:
                signal += [-1] * half
                signal += [1] * (sps - half)

    return np.array(signal, dtype=float)


# =====================================================
# PULSE SHAPING
# =====================================================

def pulse_shape(x, pulse, sps, bt, beta, span):

    if pulse == "Rectangular":
        return x.copy()

    # Filter length
    n = max(3, int(span * sps) + 1)

    if n % 2 == 0:
        n += 1

    t = np.linspace(
        -span / 2,
        span / 2,
        n
    )

    # ---------------- Gaussian ----------------

    if pulse == "Gaussian":

        sigma = np.sqrt(np.log(2)) / (
            2 * np.pi * max(bt, 0.05)
        )

        h = np.exp(
            -(t ** 2) / (2 * sigma ** 2)
        )

    # ---------------- RRC ----------------

    else:

        beta = max(0.01, min(beta, 0.99))
        h = np.zeros_like(t)

        for i, ti in enumerate(t):

            if abs(ti) < 1e-10:

                h[i] = (
                    1 - beta +
                    4 * beta / np.pi
                )

            elif abs(
                abs(4 * beta * ti) - 1
            ) < 1e-8:

                h[i] = (
                    beta / np.sqrt(2)
                ) * (
                    (1 + 2 / np.pi)
                    * np.sin(np.pi / (4 * beta))
                    +
                    (1 - 2 / np.pi)
                    * np.cos(np.pi / (4 * beta))
                )

            else:

                h[i] = (
                    np.sin(
                        np.pi * ti * (1 - beta)
                    )
                    +
                    4 * beta * ti *
                    np.cos(
                        np.pi * ti * (1 + beta)
                    )
                ) / (
                    np.pi * ti *
                    (1 - (4 * beta * ti) ** 2)
                )

    # Normalize
    s = np.sum(h)

    if abs(s) > 1e-12:
        h = h / s

    return np.convolve(x, h, mode="same")


# =====================================================
# NOISE
# =====================================================

def add_noise(x, snr, seed):

    rng = np.random.default_rng(seed)

    power = np.mean(x ** 2)

    if power < 1e-12:
        return x.copy()

    noise_power = power / (10 ** (snr / 10))

    noise = rng.normal(
        0,
        np.sqrt(noise_power),
        len(x)
    )

    return x + noise


# =====================================================
# SPECTRUM
# =====================================================

def get_spectrum(x, sps):

    x = x - np.mean(x)

    X = np.fft.rfft(x)

    f = np.fft.rfftfreq(
        len(x),
        1 / sps
    )

    mag = np.abs(X)

    if mag.max() > 0:
        mag = mag / mag.max()

    return f, mag


# =====================================================
# EYE DIAGRAM
# =====================================================

def get_eye(x, sps):

    length = 2 * sps
    traces = []

    for i in range(
        0,
        len(x) - length + 1,
        sps
    ):

        traces.append(
            x[i:i + length]
        )

        if len(traces) >= 100:
            break

    return np.array(traces)


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("Signal Settings")

    # FORM = ALL INPUTS ARE TAKEN TOGETHER
    with st.form("signal_form"):

        bits = st.text_input(
            "Binary input",
            value="10110010"
        )

        code = st.selectbox(
            "Line code",
            ["NRZ", "RZ", "Manchester"]
        )

        pulse = st.selectbox(
            "Pulse shaping",
            ["Rectangular", "Gaussian", "RRC"]
        )

        sps = st.slider(
            "Samples per bit",
            20,
            200,
            50,
            10
        )

        snr = st.slider(
            "SNR (dB)",
            0,
            30,
            20,
            1
        )

        bt = st.slider(
            "Gaussian BT",
            0.20,
            1.00,
            0.50,
            0.05
        )

        beta = st.slider(
            "RRC roll-off",
            0.10,
            0.90,
            0.35,
            0.05
        )

        span = st.slider(
            "Filter span (symbols)",
            4,
            12,
            8,
            1
        )

        seed = st.number_input(
            "Noise seed",
            min_value=0,
            max_value=999999,
            value=42,
            step=1
        )

        generate = st.form_submit_button(
            "Generate & Analyze",
            use_container_width=True
        )


# =====================================================
# GENERATE
# =====================================================

if generate:

    # Remove spaces
    bits = bits.strip().replace(" ", "")

    # Check input
    if bits == "":
        st.error("Please enter binary input.")
        st.stop()

    if any(c not in "01" for c in bits):
        st.error("Binary input must contain only 0 and 1.")
        st.stop()

    # ---------------- Line Code ----------------

    base = line_code(
        bits,
        code,
        sps
    )

    # ---------------- Pulse Shaping ----------------

    shaped = pulse_shape(
        base,
        pulse,
        sps,
        bt,
        beta,
        span
    )

    # ---------------- Noise ----------------

    received = add_noise(
        shaped,
        snr,
        int(seed)
    )

    # ---------------- Metrics ----------------

    energy = np.sum(
        received ** 2
    ) / sps

    power = np.mean(
        received ** 2
    )

    rms = np.sqrt(power)

    peak = np.max(
        np.abs(received)
    )

    # ---------------- Spectrum ----------------

    f, mag = get_spectrum(
        received,
        sps
    )

    # ---------------- Eye ----------------

    eye = get_eye(
        received,
        sps
    )

    # =================================================
    # SUCCESS
    # =================================================

    st.success(
        f"{code} + {pulse} + {snr} dB SNR generated successfully!"
    )

    # =================================================
    # SETTINGS
    # =================================================

    st.subheader("Selected Settings")

    a, b, c, d = st.columns(4)

    a.metric("Binary", bits)
    b.metric("Line Code", code)
    c.metric("Pulse", pulse)
    d.metric("SNR", f"{snr} dB")

    # =================================================
    # PLOT FUNCTION
    # =================================================

    def plot_signal(x, title):

        fig, ax = plt.subplots(
            figsize=(10, 3)
        )

        ax.plot(x)

        ax.set_title(title)
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        ax.grid(True)

        st.pyplot(fig)

        plt.close(fig)

    # =================================================
    # 1 LINE CODE
    # =================================================

    st.subheader("1. Line-Coded Signal")

    plot_signal(
        base,
        f"{code} Waveform"
    )

    # =================================================
    # 2 PULSE SHAPING
    # =================================================

    st.subheader("2. Pulse-Shaped Signal")

    plot_signal(
        shaped,
        f"{pulse} Pulse-Shaped Signal"
    )

    # =================================================
    # 3 RECEIVED
    # =================================================

    st.subheader("3. Received Signal")

    plot_signal(
        received,
        f"Received Signal - {snr} dB SNR"
    )

    # =================================================
    # 4 SPECTRUM
    # =================================================

    st.subheader("4. Frequency Spectrum")

    fig, ax = plt.subplots(
        figsize=(10, 3)
    )

    ax.plot(f, mag)

    ax.set_title(
        "Normalized Magnitude Spectrum"
    )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.grid(True)

    st.pyplot(fig)

    plt.close(fig)

    # =================================================
    # 5 METRICS
    # =================================================

    st.subheader("5. Signal Metrics")

    a, b, c, d = st.columns(4)

    a.metric(
        "Energy",
        f"{energy:.4f}"
    )

    b.metric(
        "Average Power",
        f"{power:.4f}"
    )

    c.metric(
        "RMS",
        f"{rms:.4f}"
    )

    d.metric(
        "Peak",
        f"{peak:.4f}"
    )

    # =================================================
    # 6 EYE DIAGRAM
    # =================================================

    st.subheader("6. Eye Diagram")

    if len(eye) > 0:

        fig, ax = plt.subplots(
            figsize=(8, 4)
        )

        for row in eye:
            ax.plot(
                row,
                alpha=0.3
            )

        ax.set_title(
            "Eye Diagram"
        )

        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        ax.grid(True)

        st.pyplot(fig)

        plt.close(fig)

    # =================================================
    # 7 INTERPRETATION
    # =================================================

    st.subheader("7. Interpretation")

    st.write(
        f"**{code}:** Converts 0 and 1 into a digital waveform."
    )

    st.write(
        f"**{pulse}:** Changes the shape of the transmitted pulse."
    )

    st.write(
        "**AWGN:** Adds random noise to simulate a communication channel."
    )

    st.write(
        f"**SNR = {snr} dB:** Controls the amount of noise."
    )

    st.write(
        "**Spectrum:** Shows the frequency components of the signal."
    )

    st.write(
        "**Eye Diagram:** Helps observe signal quality and timing."
    )

    # =================================================
    # 8 CSV
    # =================================================

    st.subheader("8. Download Waveform")

    df = pd.DataFrame({
        "Line_Coded": base,
        "Pulse_Shaped": shaped,
        "Received": received
    })

    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        "line_code_analyzer.csv",
        "text/csv"
    )

else:

    st.info(
        "Enter all settings on the left and click Generate & Analyze."
    )

    st.markdown(
        """
### Signal Flow

**Binary Input**
→ **Line Coding**
→ **Pulse Shaping**
→ **AWGN Noise**
→ **Received Signal**
→ **Spectrum + Metrics + Eye Diagram**
"""
    )

st.divider()

st.caption(
    "Python • NumPy • Pandas • Matplotlib • Streamlit"
)
