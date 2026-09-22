
        import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Line Code Analyzer",
    page_icon="📡",
    layout="wide"
)

st.title("📡 Line Code Analyzer")
st.caption("SP25 Digital Communication Group Project")


# ============================================================
# 1. BINARY VALIDATION
# ============================================================

def validate(bits):

    bits = bits.strip().replace(" ", "")

    if not bits:
        raise ValueError("Binary input cannot be empty.")

    if any(c not in "01" for c in bits):
        raise ValueError("Enter only 0 and 1.")

    return bits


# ============================================================
# 2. LINE CODING
# ============================================================

def line_code(bits, code, sps):

    signal = []
    half = sps // 2

    for bit in bits:

        # ---------------- NRZ ----------------
        if code == "NRZ":

            level = 1 if bit == "1" else -1

            signal += [level] * sps

        # ---------------- RZ ----------------
        elif code == "RZ":

            level = 1 if bit == "1" else -1

            signal += (
                [level] * half
                + [0] * (sps - half)
            )

        # ---------------- MANCHESTER ----------------
        else:

            if bit == "1":

                signal += (
                    [1] * half
                    + [-1] * (sps - half)
                )

            else:

                signal += (
                    [-1] * half
                    + [1] * (sps - half)
                )

    return np.array(signal, dtype=float)


# ============================================================
# 3. GAUSSIAN FILTER
# ============================================================

def gaussian_filter(sps, bt, span):

    t = np.arange(
        -span * sps / 2,
        span * sps / 2 + 1
    ) / sps

    sigma = (
        np.sqrt(np.log(2))
        / (2 * np.pi * max(bt, 0.05))
    )

    h = np.exp(
        -0.5 * (t / sigma) ** 2
    )

    # Normalize filter
    h = h / np.sum(h)

    return h


# ============================================================
# 4. RRC FILTER
# ============================================================

def rrc_filter(sps, beta, span):

    t = np.arange(
        -span * sps / 2,
        span * sps / 2 + 1,
        dtype=float
    ) / sps

    h = np.zeros(len(t))

    for i, x in enumerate(t):

        # t = 0
        if abs(x) < 1e-12:

            h[i] = (
                1
                - beta
                + 4 * beta / np.pi
            )

        # t = +/- T/(4*beta)
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

        # General case
        else:

            numerator = (
                np.sin(
                    np.pi * x * (1 - beta)
                )
                +
                4 * beta * x
                * np.cos(
                    np.pi * x * (1 + beta)
                )
            )

            denominator = (
                np.pi
                * x
                * (1 - (4 * beta * x) ** 2)
            )

            h[i] = numerator / denominator

    # Normalize filter
    h = h / np.sum(h)

    return h


# ============================================================
# 5. PULSE SHAPING
# ============================================================

def pulse_shape(
    signal,
    pulse,
    sps,
    bt,
    beta,
    span
):

    # Rectangular pulse
    if pulse == "Rectangular":

        return signal

    # Gaussian pulse
    if pulse == "Gaussian":

        h = gaussian_filter(
            sps,
            bt,
            span
        )

    # RRC pulse
    else:

        h = rrc_filter(
            sps,
            beta,
            span
        )

    # Convolution
    shaped = np.convolve(
        signal,
        h,
        mode="full"
    )

    # Keep output length same as input
    start = (len(h) - 1) // 2

    shaped = shaped[
        start:start + len(signal)
    ]

    return shaped


# ============================================================
# 6. AWGN NOISE
# ============================================================

def add_noise(signal, snr, seed):

    # Signal power
    power = np.mean(signal ** 2)

    if power <= 1e-15:

        return signal.copy()

    # Noise power
    noise_power = (
        power / (10 ** (snr / 10))
    )

    # Random generator
    rng = np.random.default_rng(
        int(seed)
    )

    # Generate Gaussian noise
    noise = rng.normal(
        0,
        np.sqrt(noise_power),
        len(signal)
    )

    # Add noise
    received = signal + noise

    return received


# ============================================================
# 7. SIGNAL METRICS
# ============================================================

def calculate_metrics(signal, fs):

    # Energy
    energy = (
        np.sum(signal ** 2)
        / fs
    )

    # Average power
    power = np.mean(
        signal ** 2
    )

    # RMS
    rms = np.sqrt(power)

    # Peak
    peak = np.max(
        np.abs(signal)
    )

    return energy, power, rms, peak


# ============================================================
# 8. FFT SPECTRUM
# ============================================================

def calculate_spectrum(signal, fs):

    # Remove DC component
    signal = signal - np.mean(signal)

    # Window
    window = np.hanning(
        len(signal)
    )

    signal = signal * window

    # FFT
    X = np.fft.rfft(signal)

    # Frequency axis
    frequency = np.fft.rfftfreq(
        len(signal),
        1 / fs
    )

    # Magnitude
    magnitude = np.abs(X)

    # Normalize
    if magnitude.max() > 0:

        magnitude = (
            magnitude
            / magnitude.max()
        )

    return frequency, magnitude


# ============================================================
# 9. EYE DIAGRAM
# ============================================================

def eye_diagram(signal, sps):

    # Two symbol periods
    length = 2 * sps

    traces = []

    # Start every symbol
    for start in range(
        0,
        len(signal) - length + 1,
        sps
    ):

        trace = signal[
            start:start + length
        ]

        traces.append(trace)

        # Maximum 120 traces
        if len(traces) == 120:

            break

    traces = np.array(traces)

    time = np.arange(length) / sps

    return traces, time


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Signal Settings")

    # Binary input
    bits = st.text_input(
        "Binary input",
        value="10110010"
    )

    # Line code
    code = st.selectbox(
        "Line code",
        [
            "NRZ",
            "RZ",
            "Manchester"
        ]
    )

    # Pulse shaping
    pulse = st.selectbox(
        "Pulse shaping",
        [
            "Rectangular",
            "Gaussian",
            "RRC"
        ]
    )

    # Samples per bit
    sps = st.slider(
        "Samples per bit",
        min_value=20,
        max_value=200,
        value=50,
        step=10
    )

    # SNR
    snr = st.slider(
        "SNR (dB)",
        min_value=0,
        max_value=30,
        value=20
    )

    # Gaussian BT
    bt = st.slider(
        "Gaussian BT",
        min_value=0.20,
        max_value=1.00,
        value=0.50,
        step=0.05
    )

    # RRC roll-off
    beta = st.slider(
        "RRC roll-off",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05
    )

    # Filter span
    span = st.slider(
        "Filter span (symbols)",
        min_value=4,
        max_value=12,
        value=8
    )

    # Noise seed
    seed = st.number_input(
        "Noise seed",
        min_value=0,
        max_value=999999,
        value=42,
        step=1
    )

    # Generate button
    generate = st.button(
        "🚀 Generate & Analyze",
        type="primary",
        use_container_width=True
    )


# ============================================================
# MAIN PROCESSING
# ============================================================

if generate:

    try:

        # ----------------------------------------------------
        # STEP 1: Validate binary input
        # ----------------------------------------------------

        bits = validate(bits)


        # ----------------------------------------------------
        # STEP 2: Line coding
        # ----------------------------------------------------

        base_signal = line_code(
            bits,
            code,
            sps
        )


        # ----------------------------------------------------
        # STEP 3: Pulse shaping
        # ----------------------------------------------------

        shaped_signal = pulse_shape(
            base_signal,
            pulse,
            sps,
            bt,
            beta,
            span
        )


        # ----------------------------------------------------
        # STEP 4: Add AWGN noise
        # ----------------------------------------------------

        received_signal = add_noise(
            shaped_signal,
            snr,
            seed
        )


        # ----------------------------------------------------
        # STEP 5: Sampling frequency
        # ----------------------------------------------------

        fs = float(sps)


        # ----------------------------------------------------
        # STEP 6: Calculate metrics
        # ----------------------------------------------------

        energy, power, rms, peak = (
            calculate_metrics(
                received_signal,
                fs
            )
        )


        # ----------------------------------------------------
        # STEP 7: Calculate FFT
        # ----------------------------------------------------

        frequency, magnitude = (
            calculate_spectrum(
                received_signal,
                fs
            )
        )


        # ----------------------------------------------------
        # STEP 8: Eye diagram
        # ----------------------------------------------------

        traces, eye_time = eye_diagram(
            received_signal,
            sps
        )


        # ====================================================
        # SUCCESS MESSAGE
        # ====================================================

        st.success(
            f"{code} + {pulse} + "
            f"{snr} dB SNR generated successfully."
        )


        # ====================================================
        # SUMMARY
        # ====================================================

        a, b, c, d = st.columns(4)

        a.metric(
            "Bits",
            bits
        )

        b.metric(
            "Line Code",
            code
        )

        c.metric(
            "Pulse",
            pulse
        )

        d.metric(
            "SNR",
            f"{snr} dB"
        )


        # ====================================================
        # 1. LINE CODED SIGNAL
        # ====================================================

        st.subheader(
            "1. Line-Coded Signal"
        )

        time = (
            np.arange(
                len(base_signal)
            ) / fs
        )

        fig, ax = plt.subplots(
            figsize=(11, 3)
        )

        ax.plot(
            time,
            base_signal
        )

        ax.set_title(
            f"{code} Waveform"
        )

        ax.set_xlabel(
            "Time (s)"
        )

        ax.set_ylabel(
            "Amplitude"
        )

        ax.grid(
            alpha=0.25
        )

        st.pyplot(
            fig,
            clear_figure=True
        )

        plt.close(fig)


        # ====================================================
        # 2. PULSE SHAPED SIGNAL
        # ====================================================

        st.subheader(
            "2. Pulse-Shaped Signal"
        )

        time = (
            np.arange(
                len(shaped_signal)
            ) / fs
        )

        fig, ax = plt.subplots(
            figsize=(11, 3)
        )

        ax.plot(
            time,
            shaped_signal
        )

        ax.set_title(
            f"{pulse} Pulse-Shaped Signal"
        )

        ax.set_xlabel(
            "Time (s)"
        )

        ax.set_ylabel(
            "Amplitude"
        )

        ax.grid(
            alpha=0.25
        )

        st.pyplot(
            fig,
            clear_figure=True
        )

        plt.close(fig)


        # ====================================================
        # 3. RECEIVED SIGNAL
        # ====================================================

        st.subheader(
            "3. Received Signal"
        )

        time = (
            np.arange(
                len(received_signal)
            ) / fs
        )

        fig, ax = plt.subplots(
            figsize=(11, 3)
        )

        ax.plot(
            time,
            received_signal
        )

        ax.set_title(
            f"Received Signal - {snr} dB SNR"
        )

        ax.set_xlabel(
            "Time (s)"
        )

        ax.set_ylabel(
            "Amplitude"
        )

        ax.grid(
            alpha=0.25
        )

        st.pyplot(
            fig,
            clear_figure=True
        )

        plt.close(fig)


        # ====================================================
        # 4. FREQUENCY SPECTRUM
        # ====================================================

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
            "Frequency (Hz)"
        )

        ax.set_ylabel(
            "Magnitude"
        )

        ax.grid(
            alpha=0.25
        )

        st.pyplot(
            fig,
            clear_figure=True
        )

        plt.close(fig)


        # ====================================================
        # 5. SIGNAL METRICS
        # ====================================================

        st.subheader(
            "5. Signal Metrics"
        )

        a, b, c, d = st.columns(4)

        a.metric(
            "Energy",
            f"{energy:.6f}"
        )

        b.metric(
            "Average Power",
            f"{power:.6f}"
        )

        c.metric(
            "RMS",
            f"{rms:.6f}"
        )

        d.metric(
            "Peak",
            f"{peak:.6f}"
        )


        # ====================================================
        # 6. EYE DIAGRAM
        # ====================================================

        st.subheader(
            "6. Eye Diagram"
        )

        if len(traces) > 0:

            fig, ax = plt.subplots(
                figsize=(8, 4)
            )

            for trace in traces:

                ax.plot(
                    eye_time,
                    trace,
                    alpha=0.35
                )

            ax.set_title(
                "Eye Diagram - 2 Symbol Intervals"
            )

            ax.set_xlabel(
                "Time (symbol periods)"
            )

            ax.set_ylabel(
                "Amplitude"
            )

            ax.grid(
                alpha=0.25
            )

            st.pyplot(
                fig,
                clear_figure=True
            )

            plt.close(fig)

        else:

            st.warning(
                "Not enough samples for eye diagram."
            )


        # ====================================================
        # 7. SIMPLE INTERPRETATION
        # ====================================================

        st.subheader(
            "7. Interpretation"
        )

        st.write(
            f"• {code} converts binary bits "
            "into a digital waveform."
        )

        st.write(
            f"• {pulse} controls the pulse shape."
        )

        st.write(
            "• AWGN simulates noise in a "
            "communication channel."
        )

        st.write(
            f"• SNR = {snr} dB. "
            "Higher SNR means less relative noise."
        )

        st.write(
            "• FFT shows the frequency content "
            "of the received signal."
        )

        st.write(
            "• Energy and power show signal strength."
        )

        st.write(
            "• RMS gives effective signal amplitude."
        )

        st.write(
            "• Peak gives maximum signal amplitude."
        )

        st.write(
            "• Eye diagram helps observe timing "
            "and signal quality."
        )


        # ====================================================
        # 8. CSV DOWNLOAD
        # ====================================================

        st.subheader(
            "8. Download Data"
        )

        df = pd.DataFrame({

            "sample":
                np.arange(
                    len(received_signal)
                ),

            "time_s":
                np.arange(
                    len(received_signal)
                ) / fs,

            "line_coded":
                base_signal,

            "pulse_shaped":
                shaped_signal,

            "received":
                received_signal
        })


        st.download_button(
            "⬇️ Download waveform CSV",

            df.to_csv(
                index=False
            ).encode(),

            "line_code_analyzer.csv",

            "text/csv"
        )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as error:

        st.error(
            f"Error: {error}"
        )


# ============================================================
# BEFORE GENERATION
# ============================================================

else:

    st.info(
        "Set the options on the left "
        "and click Generate & Analyze."
    )

    st.markdown(
        """
        ### Signal Processing Flow

        **Bits**
        ↓

        **Line Coding**
        ↓

        **Pulse Shaping**
        ↓

        **AWGN Channel**
        ↓

        **Received Signal**
        ↓

        **Frequency Spectrum**
        ↓

        **Energy / Power / RMS / Peak**
        ↓

        **Eye Diagram**
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Python • NumPy • Pandas • Matplotlib • Streamlit"
)
