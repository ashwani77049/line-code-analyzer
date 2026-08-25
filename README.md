#  SP25 Line Code Analyzer

A web app for generating and analyzing digital communication line codes.

## Features
- NRZ, RZ, Manchester
- Rectangular, Gaussian, RRC pulse shaping
- AWGN channel
- Adjustable SNR
- Time waveform
- FFT spectrum
- Energy, average power, RMS and peak
- Eye diagram
- CSV export

## Run on Windows
Open this folder in VS Code, then Terminal → New Terminal:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
```

The browser will open automatically. If not, use the local address shown in the terminal.

## First demo
Binary: `10110010`
Line code: `NRZ`
Pulse: `Rectangular`
SNR: `20 dB`

Then compare NRZ/RZ/Manchester, Rectangular/Gaussian/RRC, and SNR 30/20/10/5 dB.

## GitHub
After the app works:

```bash
git init
git add .
git commit -m "Initial Line Code Analyzer project"
git branch -M main
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

Do not upload the `venv` folder.
