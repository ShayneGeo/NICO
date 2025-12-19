import numpy as np
import os
import wave
from IPython.display import Audio, display

# ============================================================
# KICK + HI-HAT GENERATOR 
# - Two independent 16-step patterns (kick + hat)
# - Kick is a short "thump": sine wave whose pitch drops quickly + decay envelope
# - Mix both into one WAV and play inline
# ============================================================

# -------------------------
# CONFIG (EDIT THESE)
# -------------------------
OUT_DIR = r"C:\Users\"
os.makedirs(OUT_DIR, exist_ok=True)

SR = 44100
BPM = 110
bars = 4
seed = 42

# Two separate patterns (16 steps = 1 bar in 4/4 at 16ths)
kick_pattern = np.array([1,0,0,0,  0,0,1,0,  0,0,0,0,  0,1,0,0], dtype=int)
hat_pattern  = np.array([1,0,1,0,  1,0,1,0,  1,0,1,0,  1,1,0,1], dtype=int)

# Levels
kick_amp = 0.9
hat_amp  = 0.45

# Hi-hat character
hat_decay_s = 0.03

# Kick character (thump)
kick_len_s    = 0.12   # how long each kick lasts
kick_f_start  = 110.0  # starting pitch of the kick (Hz)
kick_f_end    = 45.0   # ending pitch of the kick (Hz)
kick_decay_s  = 0.08   # amplitude decay (seconds)

out_wav = os.path.join(OUT_DIR, f"kick_hat_{BPM}bpm_{bars}bars_seed{seed}.wav")

# -------------------------
# HELPERS
# -------------------------
def envelope_decay(n, sr, decay_s):
    t = np.arange(n, dtype=np.float32) / float(sr)
    return np.exp(-t / float(decay_s)).astype(np.float32)

def one_pole_lowpass(x, alpha=0.08):
    x = np.asarray(x, dtype=np.float32)
    y = np.zeros_like(x, dtype=np.float32)
    for i in range(1, len(x)):
        y[i] = (1.0 - alpha) * y[i-1] + alpha * x[i]
    return y

def save_wav(path, x, sr=44100):
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    x_i16 = (x * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(x_i16.tobytes())

def synth_hat(step_len, sr, rng, decay_s=0.03, amp=0.5):
    burst = rng.uniform(-1.0, 1.0, step_len).astype(np.float32)
    env = envelope_decay(step_len, sr, decay_s=decay_s)
    hat = burst * env * amp

    # brighten (quick high-pass)
    hat_lp = one_pole_lowpass(hat, alpha=0.08)
    hat_hp = hat - hat_lp
    return hat_hp

def synth_kick(n, sr, f_start=110.0, f_end=45.0, decay_s=0.08, amp=0.9):
    """
    Simple kick:
      - frequency drops from f_start to f_end over the hit
      - sine oscillator with exponential amplitude decay
    """
    t = np.arange(n, dtype=np.float32) / float(sr)

    # exponential-ish pitch drop
    # interpolate in log space to sound more natural
    f0 = float(f_start)
    f1 = float(f_end)
    logf = np.linspace(np.log(max(f0, 1e-6)), np.log(max(f1, 1e-6)), n).astype(np.float32)
    f_t = np.exp(logf)

    phase = 2.0 * np.pi * np.cumsum(f_t) / float(sr)
    osc = np.sin(phase).astype(np.float32)

    env = envelope_decay(n, sr, decay_s=decay_s)
    kick = osc * env * amp

    # tiny click at start (optional, gives punch)
    if n > 8:
        kick[:8] += np.linspace(0.6, 0.0, 8, dtype=np.float32) * 0.2

    return kick

# -------------------------
# BUILD THE RHYTHM GRID
# -------------------------
beats_per_sec = BPM / 60.0
steps_per_beat = 4
steps_per_sec = beats_per_sec * steps_per_beat
step_len = int(SR / steps_per_sec)

total_steps = bars * 16
total_len = total_steps * step_len

mix = np.zeros(total_len, dtype=np.float32)

rng = np.random.default_rng(seed)

# Precompute kick sample (kick lasts kick_len_s; may be longer than one step)
kick_n = int(kick_len_s * SR)
kick_sample = synth_kick(
    kick_n, SR,
    f_start=kick_f_start, f_end=kick_f_end,
    decay_s=kick_decay_s, amp=kick_amp
)

# -------------------------
# PLACE HITS ON THE GRID
# -------------------------
for s in range(total_steps):
    start = s * step_len

    # KICK
    if kick_pattern[s % 16] == 1:
        end = min(start + kick_n, total_len)
        klen = end - start
        if klen > 0:
            mix[start:end] += kick_sample[:klen]

    # HI-HAT
    if hat_pattern[s % 16] == 1:
        end = start + step_len
        if end <= total_len:
            mix[start:end] += synth_hat(step_len, SR, rng, decay_s=hat_decay_s, amp=hat_amp)

# -------------------------
# FINAL NORMALIZE + SAVE + PLAY
# -------------------------
peak = np.max(np.abs(mix)) + 1e-9
mix = (mix / peak) * 0.9  # keep some headroom

save_wav(out_wav, mix, sr=SR)
print("Saved:", out_wav)
display(Audio(mix, rate=SR))
