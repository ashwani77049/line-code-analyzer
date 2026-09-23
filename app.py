
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(page_title="Line Code Analyzer", page_icon="📡", layout="wide")

def validate_bits(bits):
    bits = bits.strip().replace(" ", "")
    if not bits or any(c not in "01" for c in bits):
        raise ValueError("Enter only binary digits 0 and 1.")
    return bits

def line_code(bits, scheme, sps):
    b = np.array([int(c) for c in bits])
    if scheme == "NRZ":
        return np.repeat(2*b-1, sps).astype(float)
    out = []
    half = sps // 2
    for bit in b:
        if scheme == "RZ":
            level = 1.0 if bit else -1.0
            out += [level]*half + [0.0]*(sps-half)
        else:  # Manchester: 1 = + then -, 0 = - then +
            a, z = (1.0, -1.0) if bit else (-1.0, 1.0)
            out += [a]*half + [z]*(sps-half)
    return np.array(out)

def gaussian_impulse(sps, bt, span):
    t = np.arange(-span*sps/2, span*sps/2+1)/sps
    sigma = np.sqrt(np.log(2))/(2*np.pi*max(bt, 0.05))
    h = np.exp(-0.5*(t/sigma)**2)
    return h/np.sum(h)

def rrc_impulse(sps, beta, span):
    beta = float(np.clip(beta, 0.01, 0.99))
    t = np.arange(-span*sps/2, span*sps/2 + 1, dtype=float) / sps
    h = np.zeros_like(t)

    for i, x in enumerate(t):
        if abs(x) < 1e-12:
            h[i] = 1 - beta + 4*beta/np.pi
        elif abs(abs(4*beta*x) - 1) < 1e-10:
            h[i] = (beta/np.sqrt(2)) * (
                (1 + 2/np.pi) * np.sin(np.pi/(4*beta))
                + (1 - 2/np.pi) * np.cos(np.pi/(4*beta))
            )
        else:
            h[i] = (
                np.sin(np.pi*x*(1-beta))
                + 4*beta*x*np.cos(np.pi*x*(1+beta))
            ) / (
                np.pi*x*(1-(4*beta*x)**2)
            )

    s = np.sum(h)

    if abs(s) > 1e-12:
        h = h / s

    return h

def shape(x, choice, sps, bt, beta, span):
    if choice == "Rectangular":
        return x.copy()

    h = gaussian_impulse(sps, bt, span) if choice == "Gaussian" else rrc_impulse(sps, beta, span)

    # Full convolution
    y = np.convolve(x, h, mode="full")

    # Keep the output centered and exactly the same length as input
    start = (len(h) - 1) // 2
    return y[start:start + len(x)]

def awgn(x, snr_db, seed):
    p = np.mean(x*x)
    if p <= 1e-15: return x.copy()
    np.random.seed(seed)
    npow = p/(10**(snr_db/10))
    return x + np.random.normal(0, np.sqrt(npow), len(x))

def metrics(x, fs):
    return np.sum(x*x)/fs, np.mean(x*x), np.sqrt(np.mean(x*x)), np.max(np.abs(x))

def spectrum(x, fs):
    x = x-np.mean(x)
    X = np.fft.rfft(x*np.hanning(len(x)))
    f = np.fft.rfftfreq(len(x),1/fs)
    m = np.abs(X); m /= m.max() if m.max() else 1
    return f,m

def eye(x, sps, max_traces=120):
    L = 2 * sps

    delay = (L - 1) // 2
    x = x[delay:]

    starts = np.arange(0, len(x) - L + 1, sps)[:max_traces]

    traces = np.array([x[s:s+L] for s in starts])

    return traces, np.arange(L) / sps

st.title(" Line Code Analyzer")
st.caption("SP25 Digital Communication Group Project")

with st.sidebar:
    st.header("Signal Settings")
    bits=st.text_input("Binary input","10110010")
    code=st.selectbox("Line code",["NRZ","RZ","Manchester"])
    pulse=st.selectbox("Pulse shaping",["Rectangular","Gaussian","RRC"])
    sps=st.slider("Samples per bit",20,200,50,10)
    snr=st.slider("SNR (dB)",0,30,20)
    bt=st.slider("Gaussian BT",.20,1.0,.50,.05,disabled=pulse!="Gaussian")
    beta=st.slider("RRC roll-off",.10,.90,.35,.05,disabled=pulse!="RRC")
    span=st.slider("Filter span (symbols)",4,12,8,1,disabled=pulse=="Rectangular")
    seed=st.number_input("Noise seed",0,999999,42)
    go=st.button(" Generate & Analyze",type="primary",use_container_width=True)

if go:
    try:
        bits=validate_bits(bits)
        fs=float(sps)
        base=line_code(bits,code,sps)
        shaped=shape(base,pulse,sps,bt,beta,span)
        received=awgn(shaped,snr,int(seed))
        energy,power,rms,peak=metrics(received,fs)
        f,mag=spectrum(received,fs)
        traces,et=eye(received,sps)
        st.session_state.result=(bits,code,pulse,sps,snr,fs,base,shaped,received,energy,power,rms,peak,f,mag,traces,et)
    except Exception as e:
        st.error(str(e))

if "result" not in st.session_state:
    st.info("Set the options on the left and click Generate & Analyze.")
    st.markdown("**Flow:** Bits → Line Coding → Pulse Shaping → AWGN → SNR → Received Signal → Spectrum + Energy/Power + Eye Diagram")
else:
    bits,code,pulse,sps,snr,fs,base,shaped,received,energy,power,rms,peak,f,mag,traces,et=st.session_state.result
    st.success(f"{code} + {pulse} + {snr} dB SNR generated successfully.")
    a,b,c,d=st.columns(4)
    a.metric("Bits",bits); b.metric("Line Code",code); c.metric("Pulse",pulse); d.metric("SNR",f"{snr} dB")

    def graph(y,title):
        fig,ax=plt.subplots(figsize=(11,3.1))
        ax.plot(np.arange(len(y))/fs,y)
        ax.set_title(title); ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude"); ax.grid(alpha=.25)
        fig.tight_layout(); st.pyplot(fig,clear_figure=True)

    st.subheader("1. Line-Coded Signal"); graph(base,f"{code} Waveform")
    st.subheader("2. Pulse-Shaped Signal"); graph(shaped,f"{pulse} Pulse-Shaped Signal")
    st.subheader("3. Received Signal"); graph(received,f"Received Signal — SNR {snr} dB")

    st.subheader("4. Frequency Spectrum")
    fig,ax=plt.subplots(figsize=(11,3.2)); ax.plot(f,mag); ax.set_title("Normalized Magnitude Spectrum"); ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Magnitude"); ax.grid(alpha=.25); fig.tight_layout(); st.pyplot(fig,clear_figure=True)

    st.subheader("5. Signal Metrics")
    a,b,c,d=st.columns(4)
    a.metric("Energy",f"{energy:.6f}"); b.metric("Average Power",f"{power:.6f}"); c.metric("RMS",f"{rms:.6f}"); d.metric("Peak",f"{peak:.6f}")

    st.subheader("6. Eye Diagram")
    if len(traces):
        fig,ax=plt.subplots(figsize=(8,4))
        for row in traces: ax.plot(et,row,alpha=.35)
        ax.set_title("Eye Diagram — 2 Symbol Intervals"); ax.set_xlabel("Time (symbol periods)"); ax.set_ylabel("Amplitude"); ax.grid(alpha=.25); fig.tight_layout(); st.pyplot(fig,clear_figure=True)

    st.subheader("7. Interpretation")
    st.write(f"- **{code}** determines how bits become a waveform.")
    st.write(f"- **{pulse}** controls pulse shape and frequency behavior.")
    st.write("- **AWGN** adds random channel noise.")
    st.write(f"- **{snr} dB SNR:** lower SNR means relatively more noise.")
    st.write("- **Spectrum** shows frequency content; energy/power quantify signal strength; the eye diagram visualizes digital signal quality.")

    df=pd.DataFrame({"sample":np.arange(len(received)),"time_s":np.arange(len(received))/fs,"line_coded":base,"pulse_shaped":shaped,"received":received})
    st.download_button(" Download waveform CSV",df.to_csv(index=False).encode(), "line_code_analyzer.csv","text/csv")

st.divider()
st.caption("Python • NumPy • SciPy • Matplotlib • Pandas • Streamlit")
