import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Line Code Analyzer",
    page_icon="📡",
    layout="wide"
)


# ================= LINE CODING =================

def line_code(bits, code, sps):

    b = [int(x) for x in bits]
    signal = []

    for bit in b:

        if code == "NRZ":
            level = 1 if bit == 1 else -1
            signal += [level] * sps

        elif code == "RZ":
            level = 1 if bit == 1 else -1
            signal += [level] * (sps // 2)
            signal += [0] * (sps - sps // 2)

        elif code == "Manchester":

            if bit == 1:
                signal += [1] * (sps // 2)
                signal += [-1] * (sps - sps // 2)

            else:
                signal += [-1] * (sps // 2)
                signal += [1] * (sps - sps // 2)

    return np.array(signal, dtype=float)


# ================= PULSE SHAPING =================

def pulse_shape(signal, pulse, sps, bt, beta, span):

    if pulse == "Rectangular":
        return signal.copy()

    # Filter time
    t = np.arange(
        -span / 2,
        span / 2 + 1 / sps,
        1 / sps
    )

    # ---------- Gaussian ----------
    if pulse == "Gaussian":

        sigma = np.sqrt(np.log(2)) / (
            2 * np.pi * max(bt, 0.05)
        )

        h = np.exp(
            -(t ** 2) / (2 * sigma ** 2)
        )

    # ---------- RRC ----------
    else:

        h = np.zeros(len(t))

        for i, x in enumerate(t):

            if abs(x) < 1e-10:

                h[i] = (
                    1 - beta +
                    4 * beta / np.pi
                )

            elif abs(abs(4 * beta * x) - 1) < 1e-10:

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
                    np.sin(np.pi * x * (1 - beta))
                    +
                    4 * beta * x
                    * np.cos(np.pi * x * (1 + beta))
                ) / (
                    np.pi * x
                    * (1 - (4 * beta * x) ** 2)
                )

    # Normalize filter
    h = h / np.sum(h)

    # Apply filter
    return np.convolve(signal, h, mode="same")


# ================= AWGN =================

def add_noise(signal, snr, seed):

    np.random.seed(seed)

    power = np.mean(signal ** 2)

    if power == 0:
        return signal

    noise_power = power / (10 ** (snr / 10))

    noise = np.random.normal(
        0,
        np.sqrt(noise_power),
        len(signal)
    )

    return signal + noise


# ================= SPECTRUM =================

def get_spectrum(signal, sps):

    signal = signal - np.mean(signal)

    X = np.fft.rfft(signal)

    frequency = np.fft.rfftfreq(
        len(signal),
        1 / sps
    )

    magnitude = np.abs(X)

    if np.max(magnitude) != 0:
        magnitude = magnitude / np.max(magnitude)

    return frequency, magnitude


# ================= EYE DIAGRAM =================

def eye_diagram(signal, sps):

    length = 2 * sps
    traces = []

    for i in range(
        0,
        len(signal) - length,
        sps
    ):

        traces.append(
            signal[i:i + length]
        )

        if len(traces) >= 100:
            break

    return np.array(traces)


# ================= SIDEBAR =================

with st.sidebar:

    st.header("Signal Settings")

    # IMPORTANT:
    # key="bits" makes input properly editable
    st.text_input(
        "Binary input",
        value="10110010",
        key="bits"
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
        min_value=20,
        max_value=200,
        value=50,
        step=10
    )

    snr = st.slider(
        "SNR (dB)",
        min_value=0,
        max_value=30,
        value=20
    )

    bt = st.slider(
        "Gaussian BT",
        min_value=0.20,
        max_value=1.00,
        value=0.50,
        step=0.05,
        disabled=(pulse != "Gaussian")
    )

    beta = st.slider(
        "RRC roll-off",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05,
        disabled=(pulse != "RRC")
    )

    span = st.slider(
        "Filter span",
        min_value=4,
        max_value=12,
        value=8,
        step=1,
        disabled=(pulse == "Rectangular")
    )

    seed = st.number_input(
        "Noise seed",
        min_value=0,
        max_value=999999,
        value=42
    )

    generate = st.button(
        "Generate & Analyze",
        type="primary",
        use_container_width=True
    )


# ================= MAIN APP =================

st.title("📡 Line Code Analyzer")

st.caption(
    "SP25 Digital Communication Group Project"
)


# ================= GENERATE =================

if generate:

    # Get current input
    bits = st.session_state.bits.strip()
    bits = bits.replace(" ", "")

    # Check binary input
    if not bits:

        st.error("Please enter binary input.")

        st.stop()

    if any(x not in "01" for x in bits):

        st.error(
            "Invalid input! Enter only 0 and 1."
        )

        st.stop()

    # ---------- Line Coding ----------

    base = line_code(
        bits,
        code,
        sps
    )

    # ---------- Pulse Shaping ----------

    shaped = pulse_shape(
        base,
        pulse,
        sps,
        bt,
        beta,
        span
    )

    # ---------- AWGN ----------

    received = add_noise(
        shaped,
        snr,
        int(seed)
    )

    # ---------- Metrics ----------

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

    # ---------- Spectrum ----------

    frequency, magnitude = get_spectrum(
        received,
        sps
    )

    # ---------- Eye Diagram ----------

    eyes = eye_diagram(
        received,
        sps
    )

    # ---------- Success ----------

    st.success(
        f"{code} + {pulse} + {snr} dB SNR generated successfully!"
    )


    # ================= INFORMATION =================

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Bits",
        bits
    )

    c2.metric(
        "Line Code",
        code
    )

    c3.metric(
        "Pulse",
        pulse
    )

    c4.metric(
        "SNR",
        f"{snr} dB"
    )


    # ================= WAVEFORM FUNCTION =================

    def show_waveform(signal, title):

        fig, ax = plt.subplots(
            figsize=(11, 3)
        )

        ax.plot(signal)

        ax.set_title(title)

        ax.set_xlabel(
            "Samples"
        )

        ax.set_ylabel(
            "Amplitude"
        )

        ax.grid(True)

        fig.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


    # ================= 1 =================

    st.subheader(
        "1. Line-Coded Signal"
    )

    show_waveform(
        base,
        f"{code} Waveform"
    )


    # ================= 2 =================

    st.subheader(
        "2. Pulse-Shaped Signal"
    )

    show_waveform(
        shaped,
        f"{pulse} Pulse-Shaped Signal"
    )


    # ================= 3 =================

    st.subheader(
        "3. Received Signal"
    )

    show_waveform(
        received,
        f"Received Signal - {snr} dB SNR"
    )


    # ================= 4 =================

    st.subheader(
        "4. Frequency Spectrum"
    )

    fig, ax = plt.subplots(
        figsize=(11, 3)
    )

    ax.plot(
        frequency,
        magnitude
    )

    ax.set_title(
        "Normalized Magnitude Spectrum"
    )

    ax.set_xlabel(
        "Frequency"
    )

    ax.set_ylabel(
        "Magnitude"
    )

    ax.grid(True)

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    # ================= 5 =================

    st.subheader(
        "5. Signal Metrics"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Energy",
        f"{energy:.4f}"
    )

    c2.metric(
        "Average Power",
        f"{power:.4f}"
    )

    c3.metric(
        "RMS",
        f"{rms:.4f}"
    )

    c4.metric(
        "Peak",
        f"{peak:.4f}"
    )


    # ================= 6 =================

    st.subheader(
        "6. Eye Diagram"
    )

    if len(eyes) > 0:

        fig, ax = plt.subplots(
            figsize=(8, 4)
        )

        for row in eyes:
            ax.plot(
                row,
                alpha=0.3
            )

        ax.set_title(
            "Eye Diagram - 2 Symbol Intervals"
        )

        ax.set_xlabel(
            "Samples"
        )

        ax.set_ylabel(
            "Amplitude"
        )

        ax.grid(True)

        fig.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


    # ================= 7 =================

    st.subheader(
        "7. Interpretation"
    )

    st.write(
        f"• **{code}** converts binary bits into a digital waveform."
    )

    st.write(
        f"• **{pulse}** changes the shape of the transmitted pulses."
    )

    st.write(
        "• **AWGN** adds random noise to simulate a communication channel."
    )

    st.write(
        f"• **{snr} dB SNR** controls the amount of noise."
    )

    st.write(
        "• Spectrum shows the frequency components of the signal."
    )

    st.write(
        "• Eye diagram helps visualize digital signal quality."
    )


    # ================= 8 =================

    st.subheader(
        "8. Download Data"
    )

    data = pd.DataFrame({
        "Line_Coded": base,
        "Pulse_Shaped": shaped,
        "Received": received
    })

    st.download_button(
        label="Download Waveform CSV",
        data=data.to_csv(index=False),
        file_name="line_code_analyzer.csv",
        mime="text/csv"
    )


# ================= BEFORE GENERATION =================

else:

    st.info(
        "Enter binary data, select the options, "
        "then click Generate & Analyze."
    )

    st.markdown(
        """
        **Signal Flow**

        Binary Bits  
        ↓  
        Line Coding  
        ↓  
        Pulse Shaping  
        ↓  
        AWGN Channel  
        ↓  
        Received Signal  
        ↓  
        Spectrum + Metrics + Eye Diagram
        """
    )


st.divider()

st.caption(
    "Python • NumPy • Pandas • Matplotlib • Streamlit"
)
