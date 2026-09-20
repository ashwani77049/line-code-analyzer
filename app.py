import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Line Code Analyzer",
    page_icon="📡",
    layout="wide"
)


# ============================================================
# 1. BINARY INPUT VALIDATION
# ============================================================

def validate_bits(bits):
    bits = bits.strip().replace(" ", "")

    if not bits:
        raise ValueError("Binary input cannot be empty.")

    if any(c not in "01" for c in bits):
        raise ValueError("Enter only binary digits 0 and 1.")

    return bits


# ============================================================
# 2. LINE CODING
# ============================================================

def line_code(bits, scheme, sps):

    b = np.array([int(c) for c in bits])

    # ---------------- NRZ ----------------
    if scheme == "NRZ":

        return np.repeat(
            2 * b - 1,
            sps
        ).astype(float)

    # ---------------- RZ / Manchester ----------------

    out = []

    half = sps // 2

    for bit in b:

        # ---------------- RZ ----------------

        if scheme == "RZ":

            level = 1.0 if bit else -1.0

            out += (
                [level] * half
                + [0.0] * (sps - half)
            )

        # ---------------- Manchester ----------------

        else:

            # 1 = +1 then -1
            # 0 = -1 then +1

            if bit == 1:
                first = 1.0
                second = -1.0

            else:
                first = -1.0
                second = 1.0

            out += (
                [first] * half
                + [second] * (sps - half)
            )

    return np.array(out)


# ============================================================
# 3. GAUSSIAN IMPULSE RESPONSE
# ============================================================

def gaussian_impulse(sps, bt, span):

    t = (
        np.arange(
            -span * sps / 2,
            span * sps / 2 + 1
        ) / sps
    )

    sigma = (
        np.sqrt(np.log(2))
        / (
            2
            * np.pi
            * max(bt, 0.05)
        )
    )

    h = np.exp(
        -0.5 * (t / sigma) ** 2
    )

    total = np.sum(h)

    if abs(total) > 1e-12:
        h = h / total

    return h


# ============================================================
# 4. RRC IMPULSE RESPONSE
# ============================================================

def rrc_impulse(sps, beta, span):

    beta = float(
        np.clip(beta, 0.01, 0.99)
    )

    t = (
        np.arange(
            -span * sps / 2,
            span * sps / 2 + 1,
            dtype=float
        ) / sps
    )

    h = np.zeros_like(t)

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
                * np.sin(
                    np.pi / (4 * beta)
                )
                +
                (1 - 2 / np.pi)
                * np.cos(
                    np.pi / (4 * beta)
                )
            )

        # General case
        else:

            numerator = (
                np.sin(
                    np.pi
                    * x
                    * (1 - beta)
                )
                +
                4
                * beta
                * x
                * np.cos(
                    np.pi
                    * x
                    * (1 + beta)
                )
            )

            denominator = (
                np.pi
                * x
                * (
                    1
                    - (4 * beta * x) ** 2
                )
            )

            h[i] = (
                numerator / denominator
            )

    total = np.sum(h)

    if abs(total) > 1e-12:
        h = h / total

    return h


# ============================================================
# 5. PULSE SHAPING
# ============================================================

def shape(
    x,
    choice,
    sps,
    bt,
    beta,
    span
):

    # Rectangular pulse
    if choice == "Rectangular":

        return x.copy()

    # Gaussian pulse
    if choice == "Gaussian":

        h = gaussian_impulse(
            sps,
            bt,
            span
        )

    # RRC pulse
    else:

        h = rrc_impulse(
            sps,
            beta,
            span
        )

    # Full convolution
    y = np.convolve(
        x,
        h,
        mode="full"
    )

    # Center the output
    start = (len(h) - 1) // 2

    return y[
        start:start + len(x)
    ]


# ============================================================
# 6. AWGN NOISE
# ============================================================

def awgn(
    x,
    snr_db,
    seed
):

    signal_power = np.mean(
        x * x
    )

    if signal_power <= 1e-15:

        return x.copy()

    # Local random generator
    rng = np.random.default_rng(
        int(seed)
    )

    noise_power = (
        signal_power
        / (
            10 ** (
                snr_db / 10
            )
        )
    )

    noise = rng.normal(
        0,
        np.sqrt(noise_power),
        len(x)
    )

    return x + noise


# ============================================================
# 7. SIGNAL METRICS
# ============================================================

def metrics(x, fs):

    energy = (
        np.sum(x * x)
        / fs
    )

    power = np.mean(
        x * x
    )

    rms = np.sqrt(
        power
    )

    peak = np.max(
        np.abs(x)
    )

    return (
        energy,
        power,
        rms,
        peak
    )


# ============================================================
# 8. FREQUENCY SPECTRUM
# ============================================================

def spectrum(x, fs):

    # Remove DC component
    x = x - np.mean(x)

    # Hanning window
    windowed = (
        x
        * np.hanning(len(x))
    )

    # FFT
    X = np.fft.rfft(
        windowed
    )

    # Frequency axis
    f = np.fft.rfftfreq(
        len(x),
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

    return (
        f,
        magnitude
    )


# ============================================================
# 9. EYE DIAGRAM
# ============================================================

def eye(
    x,
    sps,
    max_traces=120
):

    # Two symbol intervals
    L = 2 * sps

    # Centering delay
    delay = (
        L - 1
    ) // 2

    x = x[delay:]

    # Starting points
    starts = np.arange(
        0,
        len(x) - L + 1,
        sps
    )[:max_traces]

    # Extract traces
    traces = np.array([
        x[s:s + L]
        for s in starts
    ])

    # Time axis in symbol periods
    time_axis = (
        np.arange(L)
        / sps
    )

    return (
        traces,
        time_axis
    )


# ============================================================
# TITLE
# ============================================================

st.title("📡 Line Code Analyzer")

st.caption(
    "SP25 Digital Communication Group Project"
)


# ============================================================
# SIDEBAR - SIGNAL SETTINGS
# ============================================================

with st.sidebar:

    st.header("Signal Settings")


    # --------------------------------------------------------
    # Binary Input
    # --------------------------------------------------------

    bits = st.text_input(
        "Binary input",
        value="10110010",
        placeholder="Enter binary sequence"
    )


    # --------------------------------------------------------
    # Line Code
    # --------------------------------------------------------

    code = st.selectbox(
        "Line code",
        [
            "NRZ",
            "RZ",
            "Manchester"
        ]
    )


    # --------------------------------------------------------
    # Pulse Shaping
    # --------------------------------------------------------

    pulse = st.selectbox(
        "Pulse shaping",
        [
            "Rectangular",
            "Gaussian",
            "RRC"
        ]
    )


    # --------------------------------------------------------
    # Samples per Bit
    # --------------------------------------------------------

    sps = st.slider(
        "Samples per bit",
        min_value=20,
        max_value=200,
        value=50,
        step=10
    )


    # --------------------------------------------------------
    # SNR
    # --------------------------------------------------------

    snr = st.slider(
        "SNR (dB)",
        min_value=0,
        max_value=30,
        value=20,
        step=1
    )


    # --------------------------------------------------------
    # Gaussian BT
    # ALWAYS EDITABLE
    # --------------------------------------------------------

    bt = st.slider(
        "Gaussian BT",
        min_value=0.20,
        max_value=1.00,
        value=0.50,
        step=0.05
    )


    # --------------------------------------------------------
    # RRC Roll-Off
    # ALWAYS EDITABLE
    # --------------------------------------------------------

    beta = st.slider(
        "RRC roll-off",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05
    )


    # --------------------------------------------------------
    # Filter Span
    # ALWAYS EDITABLE
    # --------------------------------------------------------

    span = st.slider(
        "Filter span (symbols)",
        min_value=4,
        max_value=12,
        value=8,
        step=1
    )


    # --------------------------------------------------------
    # Noise Seed
    # --------------------------------------------------------

    seed = st.number_input(
        "Noise seed",
        min_value=0,
        max_value=999999,
        value=42,
        step=1
    )


    # --------------------------------------------------------
    # Generate Button
    # --------------------------------------------------------

    go = st.button(
        "🚀 Generate & Analyze",
        type="primary",
        use_container_width=True
    )


# ============================================================
# GENERATE & ANALYZE
# ============================================================

if go:

    try:

        # ----------------------------------------------------
        # Validate binary input
        # ----------------------------------------------------

        bits = validate_bits(
            bits
        )


        # ----------------------------------------------------
        # Sampling frequency
        # ----------------------------------------------------

        fs = float(
            sps
        )


        # ----------------------------------------------------
        # Line coding
        # ----------------------------------------------------

        base = line_code(
            bits,
            code,
            sps
        )


        # ----------------------------------------------------
        # Pulse shaping
        # ----------------------------------------------------

        shaped = shape(
            base,
            pulse,
            sps,
            bt,
            beta,
            span
        )


        # ----------------------------------------------------
        # AWGN
        # ----------------------------------------------------

        received = awgn(
            shaped,
            snr,
            int(seed)
        )


        # ----------------------------------------------------
        # Signal metrics
        # ----------------------------------------------------

        (
            energy,
            power,
            rms,
            peak
        ) = metrics(
            received,
            fs
        )


        # ----------------------------------------------------
        # Spectrum
        # ----------------------------------------------------

        (
            f,
            mag
        ) = spectrum(
            received,
            fs
        )


        # ----------------------------------------------------
        # Eye diagram
        # ----------------------------------------------------

        (
            traces,
            et
        ) = eye(
            received,
            sps
        )


        # ----------------------------------------------------
        # Store everything
        # ----------------------------------------------------

        st.session_state.result = (

            bits,
            code,
            pulse,
            sps,
            snr,
            fs,
            base,
            shaped,
            received,
            energy,
            power,
            rms,
            peak,
            f,
            mag,
            traces,
            et
        )


    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )


# ============================================================
# BEFORE GENERATION
# ============================================================

if "result" not in st.session_state:

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
# RESULTS
# ============================================================

else:

    (
        bits,
        code,
        pulse,
        sps,
        snr,
        fs,
        base,
        shaped,
        received,
        energy,
        power,
        rms,
        peak,
        f,
        mag,
        traces,
        et
    ) = st.session_state.result


    # ========================================================
    # SUCCESS MESSAGE
    # ========================================================

    st.success(
        f"{code} + {pulse} + "
        f"{snr} dB SNR generated successfully."
    )


    # ========================================================
    # SUMMARY
    # ========================================================

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


    # ========================================================
    # GRAPH FUNCTION
    # ========================================================

    def graph(
        y,
        title
    ):

        fig, ax = plt.subplots(
            figsize=(11, 3.1)
        )

        ax.plot(
            np.arange(len(y)) / fs,
            y
        )

        ax.set_title(
            title
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

        fig.tight_layout()

        st.pyplot(
            fig,
            clear_figure=True
        )


    # ========================================================
    # 1. LINE-CODED SIGNAL
    # ========================================================

    st.subheader(
        "1. Line-Coded Signal"
    )

    graph(
        base,
        f"{code} Waveform"
    )


    # ========================================================
    # 2. PULSE-SHAPED SIGNAL
    # ========================================================

    st.subheader(
        "2. Pulse-Shaped Signal"
    )

    graph(
        shaped,
        f"{pulse} Pulse-Shaped Signal"
    )


    # ========================================================
    # 3. RECEIVED SIGNAL
    # ========================================================

    st.subheader(
        "3. Received Signal"
    )

    graph(
        received,
        f"Received Signal — SNR {snr} dB"
    )


    # ========================================================
    # 4. FREQUENCY SPECTRUM
    # ========================================================

    st.subheader(
        "4. Frequency Spectrum"
    )

    fig, ax = plt.subplots(
        figsize=(11, 3.2)
    )

    ax.plot(
        f,
        mag
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

    fig.tight_layout()

    st.pyplot(
        fig,
        clear_figure=True
    )


    # ========================================================
    # 5. SIGNAL METRICS
    # ========================================================

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


    # ========================================================
    # 6. EYE DIAGRAM
    # ========================================================

    st.subheader(
        "6. Eye Diagram"
    )

    if len(traces) > 0:

        fig, ax = plt.subplots(
            figsize=(8, 4)
        )

        for row in traces:

            ax.plot(
                et,
                row,
                alpha=0.35
            )

        ax.set_title(
            "Eye Diagram — 2 Symbol Intervals"
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

        fig.tight_layout()

        st.pyplot(
            fig,
            clear_figure=True
        )

    else:

        st.warning(
            "Not enough samples to generate "
            "the eye diagram."
        )


    # ========================================================
    # 7. INTERPRETATION
    # ========================================================

    st.subheader(
        "7. Interpretation"
    )

    st.write(
        f"• **{code}** determines how binary bits "
        "are converted into a digital waveform."
    )

    st.write(
        f"• **{pulse}** determines the pulse shape "
        "and affects the frequency characteristics."
    )

    st.write(
        "• **AWGN** adds random noise to simulate "
        "a communication channel."
    )

    st.write(
        f"• **{snr} dB SNR:** lower SNR means "
        "relatively more noise."
    )

    st.write(
        "• **Spectrum** shows the frequency content "
        "of the received signal."
    )

    st.write(
        "• **Energy and Power** quantify signal strength."
    )

    st.write(
        "• **RMS** represents the effective signal amplitude."
    )

    st.write(
        "• **Peak** represents the maximum absolute amplitude."
    )

    st.write(
        "• **Eye Diagram** helps visualize timing "
        "and digital signal quality."
    )


    # ========================================================
    # CSV DOWNLOAD
    # ========================================================

    df = pd.DataFrame({

        "sample":
            np.arange(
                len(received)
            ),

        "time_s":
            np.arange(
                len(received)
            ) / fs,

        "line_coded":
            base,

        "pulse_shaped":
            shaped,

        "received":
            received
    })


    st.download_button(
        "⬇️ Download waveform CSV",

        df.to_csv(
            index=False
        ).encode(),

        "line_code_analyzer.csv",

        "text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Python • NumPy • Pandas • Matplotlib • Streamlit"
)
