"""Make the Version 2 breathing audio (4 s in / 6 s out paced breathing, and the physiological-sigh pattern), 5 minutes each.

Each phase starts with a spoken cue ("breathe in" / "breathe out", macOS `say`), under a soft tone that rises in pitch
during the inhale and falls during the exhale. Paced breathing timing (4 s in, 6 s out) follows You et al. 2021.
The SIGH timings (3.0 s in, 1.2 s second short in, 6.0 s out) are NOT from a paper: they are our own labeled design choice.
No data files are read. Usage: .venv/bin/python src/make_audio.py
"""
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "audio"
SR = 22050
TOTAL_S = 300
F_LOW, F_HIGH = 200.0, 300.0        # tone pitch range (Hz), our choice
TONE_VOL, VOICE_VOL = 0.18, 0.8


def speak(text, tmp):
    aiff, wav = Path(tmp) / "v.aiff", Path(tmp) / "v.wav"
    subprocess.run(["say", "-v", "Samantha", "-r", "140", "-o", str(aiff), text], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{SR}", "-c", "1", str(aiff), str(wav)], check=True)
    with wave.open(str(wav)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(float) / 32768
    return x / max(1e-9, np.abs(x).max())


def sweep(f0, f1, dur):
    n = int(dur * SR)
    f = np.linspace(f0, f1, n)
    phase = 2 * np.pi * np.cumsum(f) / SR
    env = np.minimum(1, np.minimum(np.arange(n), n - 1 - np.arange(n)) / (0.15 * SR))   # 0.15 s fade in/out
    return np.sin(phase) * env * TONE_VOL


def phase(tone, voice, dur):
    seg = tone.copy()
    v = voice[: len(seg)]
    seg[: len(v)] += v * VOICE_VOL
    return seg


def write(name, cycle_parts):
    cycle = np.concatenate(cycle_parts)
    n_cycles = int(TOTAL_S // (len(cycle) / SR))
    x = np.concatenate([cycle] * n_cycles)
    x = np.concatenate([x, np.zeros(TOTAL_S * SR - len(x))])
    x = np.clip(x, -1, 1)
    path = OUT / name
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((x * 32767).astype("<i2").tobytes())
    print(f"{path.name}: {n_cycles} cycles of {len(cycle) / SR:.1f} s, {len(x) / SR:.0f} s total")


def main():
    OUT.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        v_in, v_out, v_again = speak("Breathe in", tmp), speak("Breathe out", tmp), speak("In again", tmp)
    write("paced_breathing_4in_6out.wav", [
        phase(sweep(F_LOW, F_HIGH, 4.0), v_in, 4.0),
        phase(sweep(F_HIGH, F_LOW, 6.0), v_out, 6.0)])
    write("physiological_sigh.wav", [
        phase(sweep(F_LOW, F_HIGH * 0.9, 3.0), v_in, 3.0),
        phase(sweep(F_HIGH * 0.9, F_HIGH, 1.2), v_again, 1.2),
        phase(sweep(F_HIGH, F_LOW, 6.0), v_out, 6.0)])


if __name__ == "__main__":
    main()
