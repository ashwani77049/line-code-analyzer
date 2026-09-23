import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config("Line Code Analyzer", "📡", layout="wide")

# ---------- LINE CODING ----------
def line_code(bits, code, sps):
    b = np.array([int(x) for x in bits])
    out = []

    for bit in b:
        if code == "NRZ":
            out += [1 if bit else -1] * sps

        elif code == "RZ":
            out += ([1 if bit else -1] * (sps//2)) + [0] * (sps//2)

        else:  # Manchester
            if bit:
                out += [1] * (sps//2) + [-1] * (sps//2)
            else:
                out += [-1] * (sps//2) + [1] * (sps//2)

    return np.array(out, dtype=float)


# ---------- PULSE SHAPING ----------
def pulse_shape(x, pulse, sps, bt, beta, span):

    if pulse == "Rectangular":
        return x

    t = np.arange(-span/2, span/2 + 1/sps, 1/sps)

    if pulse == "Gaussian":
        sigma = np.sqrt(np.log(2)) / (2*np.pi*max(bt, 0.05))
        h = np.exp(-t*t/(2*sigma*sigma))

    else:  # RRC
        h = np.zeros(len(t))

        for i, x1 in enumerate(t):
            if abs(x1) < 1e-10:
                h[i] = 1 - beta + 4*beta/np.pi

            elif abs(abs(4*beta*x1)-1) < 1e-10:
                h[i] = (beta/np.sqrt(2))*(
                    (1+2/np.pi)*np.sin(np.pi/(4*beta)) +
                    (1-2/np.pi)*np.cos(np.pi/(4*beta))
                )

            else:
                h[i] = (
                    np.sin(np.pi*x1*(1-beta)) +
                    4*beta*x1*np.cos(np.pi*x1*(1+beta))
                ) / (
                    np.pi*x1*(1-(4*beta*x1)**2)
                )

    h = h / np.sum(h)

    y = np.convolve(x, h, mode="same")
    return y


# ---------- NOISE ----------
def add_noise(x, snr, seed):
    np.random.seed(seed)

    power = np.mean(x*x)
    noise_power = power / (10**(snr/10))

    return x + np.random.normal(
        0, np.sqrt(noise_power), len(x)
    )


# ---------- SPECTRUM ----------
def spectrum(x, fs):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1/fs)

    m = abs(X)
    m = m / max(m.max(), 1)

    return f, m


# ---------- EYE DIAGRAM ----------
def eye_diagram(x, sps):
    L = 2*sps
    traces = []

    for i in range(0, len(x)-L, sps):
        traces.append(x[i:i+L])

    return np.array(traces[:100])


# ================= APP =================

st.title("📡 Line Code Analyzer")
st.caption("SP25 Digital Communication Project")

# ---------- SIDEBAR ----------
with st.sidebar:

    st.header("Signal Settings")

    bits = st.text_input(
        "Binary input",
        "10110010"
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
        20, 200, 50, 10
    )

    snr = st.slider(
        "SNR (dB)",
        0, 30, 20
    )

    bt = st.slider(
        "Gaussian BT",
        0.2, 1.0, 0.5, 0.05,
        disabled=(pulse != "Gaussian")
    )

    beta = st.slider(
        "RRC roll-off",
        0.1, 0.9, 0.35, 0.05,
        disabled=(pulse != "RRC")
    )

    span = st.slider(
        "Filter span",
        4, 12, 8,
        disabled=(pulse == "Rectangular")
    )

    seed = st.number_input(
        "Noise seed",
        0, 999999, 42
    )

    run = st.button(
        "Generate & Analyze",
        type="primary",
        use_container_width=True
    )


# ---------- GENERATE ----------
if run:

    if not bits or any(c not in "01" for c in bits):
        st.error("Enter only 0 and 1.")
        st.stop()

    base = line_code(bits, code, sps)

    shaped = pulse_shape(
        base, pulse, sps, bt, beta, span
    )

    received = add_noise(
        shaped, snr, seed
    )

    # Metrics
    energy = np.sum(received**2) / sps
    power = np.mean(received**2)
    rms = np.sqrt(power)
    peak = np.max(abs(received))

    # Spectrum
    f, mag = spectrum(received, sps)

    st.success(
        f"{code} + {pulse} + {snr} dB SNR generated!"
    )

    # ---------- WAVEFORMS ----------
    def plot_signal(signal, title):
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(signal)
        ax.set_title(title)
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        ax.grid()
        st.pyplot(fig)

    st.subheader("1. Line-Coded Signal")
    plot_signal(base, code + " Waveform")

    st.subheader("2. Pulse-Shaped Signal")
    plot_signal(shaped, pulse + " Waveform")

    st.subheader("3. Received Signal")
    plot_signal(received, "Received Signal")

    # ---------- SPECTRUM ----------
    st.subheader("4. Frequency Spectrum")

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(f, mag)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Magnitude")
    ax.grid()
    st.pyplot(fig)

    # ---------- METRICS ----------
    st.subheader("5. Signal Metrics")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Energy", f"{energy:.4f}")
    c2.metric("Power", f"{power:.4f}")
    c3.metric("RMS", f"{rms:.4f}")
    c4.metric("Peak", f"{peak:.4f}")

    # ---------- EYE ----------
    st.subheader("6. Eye Diagram")

    traces = eye_diagram(received, sps)

    if len(traces):
        fig, ax = plt.subplots(figsize=(8, 4))

        for row in traces:
            ax.plot(row, alpha=0.3)

        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        ax.grid()

        st.pyplot(fig)

    # ---------- CSV ----------
    st.subheader("7. Download Data")

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
        "Set the options and click Generate & Analyze."
    )

    st.write(
        "**Flow:** Bits → Line Coding → Pulse Shaping "
        "→ AWGN → Received Signal → Spectrum + Metrics + Eye Diagram"
    )
