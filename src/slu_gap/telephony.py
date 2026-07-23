"""Telephony channel simulation for synthetic speech.

Post-processes studio-quality TTS output to match the acoustic profile of the
real Skit-S2I corpus: 8 kHz, band-limited, G.711 mu-law, recorded over a phone
line. Every constant here was calibrated against measurements of the real audio
(see experiments/diagnostics/skit_s2i_audio_characterization.py), and the chain
matches the real spectrum to within about 7% on every metric we measured.

The paper reports this as a NEGATIVE result and it is worth stating plainly:
matching the channel this closely moved synthetic-only transfer accuracy by
only +0.93pp. Channel mismatch is not what makes synthetic speech hard to
transfer -- speaker identity is. This module exists because ruling the channel
out was necessary to establish that.

Stages, in order:
  1. resample to 8 kHz (anti-aliased polyphase)
  2. band-pass 250-3700 Hz (2nd-order Butterworth, zero-phase)
  3. single-tap reverb (handset coloration)
  4. RMS-normalise to the real median
  5. soft tanh peak limit
  6. pink noise floor + 60 Hz line hum
  7. occasional packet-loss dropout
  8. int16 PCM quantisation
  9. mu-law (G.711) encode/decode round-trip
"""

import audioop

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

TELEPHONE_SR = 8000

# Telephone band-pass cutoffs (Hz). Tightened from 200/3700 to 250/3700 after a
# smoke-test showed 200 Hz was overshooting the real low-end share. Real has
# ~14% energy <300 Hz, 82% in 300-3400, 0% >3400; 250-3700 with a 2nd-order
# filter produces a similar distribution without the sharp rolloff of the old
# 4th-order 300-3400 Butterworth (which suppressed real's natural low-end).
TELEPHONE_BAND_LOW = 250.0
TELEPHONE_BAND_HIGH = 3700.0
# Noise/reverb knobs. Real Skit-S2I has an extremely quiet noise floor in
# pauses (~0.0001 RMS, ~ -75 dB below speech). Tuned via smoke v2:
# v2 produced noise floor 0.0114 with -55 dB / 0.18 reverb gain — reverb tail
# and pink noise were both too loud, so we drop noise to -75 dB, hum to -90 dB,
# and reverb gain to 0.08 with 1 tap (subtle handset coloration, not echo).
TELEPHONY_NOISE_DB_BELOW_SPEECH = -75.0   # pink noise floor (telephony hiss)
TELEPHONY_HUM_DB_BELOW_SPEECH = -90.0     # 60 Hz line hum (barely perceptible)
TELEPHONY_REVERB_DELAY_MS = 20
TELEPHONY_REVERB_TAPS = 1
TELEPHONY_REVERB_GAIN = 0.08
TELEPHONY_PACKETLOSS_PROB = 0.08          # 8% of utterances get one dropout
TELEPHONY_PACKETLOSS_MS_RANGE = (20, 45)
# Real Skit-S2I median overall RMS is 0.077 and peak ~0.73. We normalize overall
# RMS (not p90 speech RMS) because that's the steadiest metric and matches the
# real recording target. Peak limiter caps at 0.73 to avoid clipping.
TELEPHONY_TARGET_RMS = 0.077
TELEPHONY_PEAK_LIMIT = 0.73

def _pink_noise(n, rng):
    """Pink (1/f) noise via FFT-domain spectral shaping. Returns unit-RMS array."""
    white = rng.standard_normal(n).astype(np.float32)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    shape = np.ones_like(freqs)
    shape[1:] = 1.0 / np.sqrt(freqs[1:])
    spec = spec * shape
    pink = np.fft.irfft(spec, n=n).astype(np.float32)
    std = float(np.std(pink))
    return pink / std if std > 1e-9 else pink


def _apply_simple_reverb(audio, target_sr, delay_ms, n_taps, gain):
    """Multi-tap feedback delay reverb (speakerphone-grade). Pre-allocated for speed."""
    out = audio.copy()
    delay_samples = int(delay_ms * target_sr / 1000)
    for i in range(1, n_taps + 1):
        d = delay_samples * i
        if d >= len(out):
            break
        out[d:] += audio[: len(out) - d] * (gain ** i)
    return out


def apply_telephony_postprocess(audio_f32, src_sr, target_sr=TELEPHONE_SR, seed=None):
    """Resample to 8 kHz, gentle band-pass, mild reverb, telephony noise floor,
    occasional packet-loss dropout, mu-law encode/decode.

    Tuned to the real Skit-S2I audio profile in
    analysis/skit_s2i_audio_characterization.json.

    Args:
        audio_f32: 1-D float32 audio array, range roughly [-1, 1]
        src_sr: original sample rate
        target_sr: telephony sample rate (8000)
        seed: optional int for reproducibility of noise / packet loss per utterance

    Returns:
        int16 PCM array at target_sr.
    """
    rng = np.random.default_rng(seed)

    # 1. Resample to telephony SR (anti-aliased polyphase filter)
    if src_sr != target_sr:
        from math import gcd
        g = gcd(src_sr, target_sr)
        up = target_sr // g
        down = src_sr // g
        audio = resample_poly(audio_f32, up=up, down=down).astype(np.float32)
    else:
        audio = audio_f32.astype(np.float32)

    # 2. Gentler band-pass 200-3700 Hz, 2nd-order Butterworth.
    sos = butter(
        N=2,
        Wn=[TELEPHONE_BAND_LOW, TELEPHONE_BAND_HIGH],
        btype="bandpass",
        fs=target_sr,
        output="sos",
    )
    audio = sosfiltfilt(sos, audio).astype(np.float32)

    # 3. Mild reverb (speakerphone character).
    audio = _apply_simple_reverb(
        audio, target_sr,
        TELEPHONY_REVERB_DELAY_MS, TELEPHONY_REVERB_TAPS, TELEPHONY_REVERB_GAIN,
    )

    # 4. Loudness-normalize overall RMS to the real Skit-S2I median (0.077).
    # Using overall RMS (not p90 windowed RMS) because reverb tail and pink
    # noise reaching the p10/p90 windows skews if we target speech_rms directly.
    current_rms = float(np.sqrt(np.mean(audio ** 2)) + 1e-9)
    if current_rms > 1e-6:
        audio = audio * (TELEPHONY_TARGET_RMS / current_rms)

    # 5. Soft peak limiter at 0.73 (real Skit-S2I median peak). Soft-clip via
    # tanh to avoid hard-clip distortion when normalization left some loud peaks.
    peak = float(np.abs(audio).max())
    if peak > TELEPHONY_PEAK_LIMIT:
        # Smooth compression: tanh maps (-inf, inf) -> (-1, 1); scale so the
        # current peak lands at TELEPHONY_PEAK_LIMIT.
        audio = np.tanh(audio / TELEPHONY_PEAK_LIMIT * 0.95).astype(np.float32) * TELEPHONY_PEAK_LIMIT

    # 6. Add pink noise + 60 Hz hum at calibrated levels below the target RMS.
    # Levels are referenced to TELEPHONY_TARGET_RMS so they remain stable across
    # utterances regardless of original TTS loudness.
    ref_rms = TELEPHONY_TARGET_RMS
    noise_rms = ref_rms * (10 ** (TELEPHONY_NOISE_DB_BELOW_SPEECH / 20.0))
    pink = _pink_noise(len(audio), rng) * noise_rms
    hum_rms = ref_rms * (10 ** (TELEPHONY_HUM_DB_BELOW_SPEECH / 20.0))
    t = np.arange(len(audio), dtype=np.float32) / target_sr
    hum = (np.sin(2 * np.pi * 60.0 * t).astype(np.float32) * hum_rms)
    audio = audio + pink + hum

    # 7. Occasional packet-loss dropout.
    if rng.random() < TELEPHONY_PACKETLOSS_PROB and len(audio) > target_sr // 4:
        lo_ms, hi_ms = TELEPHONY_PACKETLOSS_MS_RANGE
        dropout_ms = int(rng.integers(lo_ms, hi_ms + 1))
        dropout_samples = int(dropout_ms * target_sr / 1000)
        usable = len(audio) - dropout_samples
        if usable > 0:
            # avoid dropping the very first/last 20% (preserves intent words)
            lo_idx = int(0.2 * len(audio))
            hi_idx = max(lo_idx + 1, int(0.8 * len(audio)) - dropout_samples)
            if hi_idx > lo_idx:
                start = int(rng.integers(lo_idx, hi_idx))
                audio[start:start + dropout_samples] = 0.0

    # 8. Convert float [-1, 1] to int16 PCM
    audio = np.clip(audio, -1.0, 1.0)
    pcm_int16 = (audio * 32767.0).astype(np.int16)

    # 9. mu-law encode + decode for G.711 codec artifacts
    pcm_bytes = pcm_int16.tobytes()
    ulaw_bytes = audioop.lin2ulaw(pcm_bytes, 2)  # 2 = sample width in bytes (int16)
    decoded_bytes = audioop.ulaw2lin(ulaw_bytes, 2)
    pcm_int16 = np.frombuffer(decoded_bytes, dtype=np.int16).copy()

    return pcm_int16
