import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Line Code Analyzer", page_icon="📡", layout="wide")
st.title("📡 Line Code Analyzer")


# ---------- LINE CODING ----------

def line_code(bits, mode, sps):
    x = []

    for b in bits:
        v = 1 if b == "1" else -1

        if mode == "NRZ":
            x += [v] * sps

        elif mode == "RZ":
            x += [v] * (sps // 2)
            x += [0] * (sps - sps // 2)

        else:                       # Manchester
            x += [v] * (sps // 2)
            x += [-v] * (sps - sps // 2)

    return np.array(x, dtype=float)


# ---------- RRC ----------

def rrc(sps, beta=0.35, span=6):
    t = np.arange(-span*sps/2, span*sps/2 + 1) / sps
    h = np.zeros(len(t))

    for i, x in enumerate(t):
        if abs(x) < 1e-10:
            h[i] = 1 + beta * (4/np.pi - 1)

        elif abs(abs(x) - 1/(4*beta)) < 1e-10:
            h[i] = (beta/np.sqrt(2)) * (
                (1 + 2/np.pi) * np.sin(np.pi/(4*beta))
                + (1 - 2/np.pi) * np.cos(np.pi/(4*beta))
            )

        else:
            num = (
                np.sin(np.pi*x*(1-beta))
                + 4*beta*x*np.cos(np.pi*x*(1+beta))
            )
            den = np.pi*x*(1-(4*beta*x)**2)
            h[i] = num / den

    return h / np.sqrt(np.sum(h**2))


# ---------- PULSE SHAPING ----------

def shape_signal(x, mode, sps, beta, span):

    if mode == "Rectangular":
        return x.copy()

    if mode == "Gaussian":
        t = np.linspace(-2, 2, 2*sps + 1)
        p = np.exp(-(t**2)/(2*0.7**2))
        p /= np.sum(p)

    else:
        p = rrc(sps, beta, span)

    y = np.convolve(x, p, mode="same")
    y /= np.max(np.abs(y))

    return y


# ---------- NOISE ----------

def add_noise(x, snr, seed):
    rng = np.random.default_rng(seed)

    power = np.mean(x**2)
    noise_power = power / (10**(snr/10))

    noise = rng.normal(
        0,
        np.sqrt(noise_power),
        len(x)
    )

    return x + noise, noise


# ---------- METRICS ----------

def metrics(x):
    power = np.mean(x**2)

    return (
        np.sum(x**2),
        power,
        np.sqrt(power),
        np.max(np.abs(x))
    )


# ---------- FFT ----------

def spectrum(x):
    y = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x))

    return f, np.abs(y) / len(x)


# ---------- EYE DIAGRAM ----------

def eye(x, sps):
    n = min(100, len(x)//sps - 1)

    if n <= 0:
        return None

    return np.array([
        x[i*sps:i*sps+2*sps]
        for i in range(n)
        if len(x[i*sps:i*sps+2*sps]) == 2*sps
    ])


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Settings")

sps = st.sidebar.slider(
    "Samples per bit",
    20, 100, 50, 10
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
    0, 30, 20
)

seed = st.sidebar.number_input(
    "Noise Seed",
    0, 10000, 42
)

if pulse == "RRC":
    beta = st.sidebar.slider(
        "RRC Roll-off β",
        0.1, 1.0, 0.35, 0.05
    )

    span = st.sidebar.slider(
        "RRC Span",
        2, 10, 6
    )
else:
    beta = 0.35
    span = 6


# =========================================================
# INPUT
# =========================================================

st.subheader("1️⃣ Binary Input")

bits = st.text_input(
    "Enter binary data",
    "10110010"
)

st.caption("Only 0 and 1 are allowed.")


if st.button("🚀 Generate Signal", type="primary"):

    # Input checking
    bits = bits.replace(" ", "").strip()

    if not bits:
        st.error("Please enter binary data.")
        st.stop()

    if any(b not in "01" for b in bits):
        st.error("Invalid input. Use only 0 and 1.")
        st.stop()

    if len(bits) > 2000:
        st.error("Please use 2000 bits or fewer.")
        st.stop()


    # ---------- SIGNAL GENERATION ----------

    coded = line_code(
        bits,
        coding,
        sps
    )

    shaped = shape_signal(
        coded,
        pulse,
        sps,
        beta,
        span
    )

    received, noise = add_noise(
        shaped,
        snr,
        seed
    )

    time = np.arange(len(coded)) / sps


    st.success("Signal generated successfully!")

    st.write(
        f"**Input:** {bits}  |  "
        f"**Bits:** {len(bits)}  |  "
        f"**Coding:** {coding}  |  "
        f"**Pulse:** {pulse}  |  "
        f"**SNR:** {snr} dB"
    )


    # =====================================================
    # LINE CODE
    # =====================================================

    st.subheader("2️⃣ Line-Coded Signal")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time, coded)
    ax.set_title(coding)
    ax.set_xlabel("Bit Time")
    ax.set_ylabel("Amplitude")
    ax.grid()
    st.pyplot(fig)
    plt.close(fig)


    # =====================================================
    # PULSE SHAPING
    # =====================================================

    st.subheader("3️⃣ Pulse-Shaped Signal")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time, shaped)

    title = pulse + " Pulse Shaping"

    if pulse == "RRC":
        title += f"  (β={beta}, Span={span})"

    ax.set_title(title)
    ax.set_xlabel("Bit Time")
    ax.set_ylabel("Amplitude")
    ax.grid()

    st.pyplot(fig)
    plt.close(fig)


    # =====================================================
    # NOISE
    # =====================================================

    st.subheader("4️⃣ Channel Noise")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time, noise)
    ax.set_title("AWGN Noise")
    ax.set_xlabel("Bit Time")
    ax.set_ylabel("Amplitude")
    ax.grid()

    st.pyplot(fig)
    plt.close(fig)


    # =====================================================
    # RECEIVED SIGNAL
    # =====================================================

    st.subheader("5️⃣ Received Signal")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time, received)
    ax.set_title(f"Received Signal — SNR = {snr} dB")
    ax.set_xlabel("Bit Time")
    ax.set_ylabel("Amplitude")
    ax.grid()

    st.pyplot(fig)
    plt.close(fig)


    # =====================================================
    # FFT
    # =====================================================

    st.subheader("6️⃣ Frequency Spectrum")

    f, mag = spectrum(received)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(f, mag)
    ax.set_title("FFT Magnitude Spectrum")
    ax.set_xlabel("Normalized Frequency")
    ax.set_ylabel("Magnitude")
    ax.grid()

    st.pyplot(fig)
    plt.close(fig)


    # =====================================================
    # METRICS
    # =====================================================

    st.subheader("7️⃣ Signal Measurements")

    energy, power, rms, peak = metrics(received)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Energy", f"{energy:.3f}")
    c2.metric("Average Power", f"{power:.3f}")
    c3.metric("RMS", f"{rms:.3f}")
    c4.metric("Peak", f"{peak:.3f}")


    # =====================================================
    # EYE DIAGRAM
    # =====================================================

    st.subheader("8️⃣ Eye Diagram")

    e = eye(received, sps)

    if e is not None:

        fig, ax = plt.subplots(figsize=(10, 5))

        t_eye = np.arange(2*sps) / sps

        for row in e:
            ax.plot(t_eye, row, alpha=0.3)

        ax.set_title("Eye Diagram")
        ax.set_xlabel("Time (Bit Periods)")
        ax.set_ylabel("Amplitude")
        ax.grid()

        st.pyplot(fig)
        plt.close(fig)

    else:
        st.warning("Not enough samples for eye diagram.")


    # =====================================================
    # CSV
    # =====================================================

    st.subheader("9️⃣ Download Results")

    data = pd.DataFrame({
        "Time": time,
        "Line_Coded": coded,
        "Pulse_Shaped": shaped,
        "Noise": noise,
        "Received": received
    })

    st.download_button(
        "⬇️ Download CSV",
        data.to_csv(index=False),
        "line_code_analysis.csv",
        "text/csv"
    )
