"""
realistic_dummy_websocket.py  (updated)
========================================
Simulates the tubing manufacturing process at 2400 Hz.

Key change from previous version
─────────────────────────────────
Chatter onset is now a *strong function of normalised section progress*
(footage / target_footage), rather than just air-ramp pressure.  This
makes the pressure/footage → chatter relationship obvious enough to train
and test HybridChatterNet's context branch in a short session:

  progress < 0.30   →  chatter almost impossible  (≤ 0.1 % / second)
  progress 0.30–0.55 →  rapidly growing probability
  progress 0.55–0.75 →  moderate–high  (20–50 % / second)
  progress > 0.75   →  near-certain   (60–85 % / second)

Chatter amplitude and burst duration also scale with progress, so the
OD signal itself becomes visibly noisier in the final quarter of a section.

TIME_SCALE compresses section time for faster testing:
  TIME_SCALE = 1.0  → real time  (~15–40 min per section)
  TIME_SCALE = 10.0 → 10× faster (~1.5–4 min per section)  ← default
"""

import asyncio
import json
import math
import random
import time
from datetime import datetime, timezone

import numpy as np
import websockets

# ─────────────────────────────── MASTER CONFIG ───────────────────────────────

TIME_SCALE  = 20.0       # section-progress compression (does NOT alter 2400 Hz OD rate)
fs          = 2400.0     # OD sensor sample rate [Hz]
dt_target   = 1.0 / fs
BATCH_SIZE  = 40         # samples per WebSocket message → ~60 msg/s

# ── OD / geometry ─────────────────────────────────────────────────────────────
MEAN_OD     = 0.069      # inches
OD_VARIANCE = 0.005      # ± natural variation across sections

# Chatter harmonics (spatial)
CHATTER_WAVELENGTHS = [1.0, 2.0]
base_amp             = 0.002 * (MEAN_OD / 0.069)
NORMAL_AMPS  = [0.6 * base_amp, 0.7 * base_amp]   # quiet running
CHATTER_AMPS = [8.0 * base_amp, 6.0 * base_amp]   # base chatter amplitude

NOISE_RUNNING = 0.0002 * (MEAN_OD / 0.069)
NOISE_STOPPED = 0.00002 * (MEAN_OD / 0.069)
NOISE_CHATTER = 0.003   * (MEAN_OD / 0.069)

# ── Speed ─────────────────────────────────────────────────────────────────────
RUN_SPEED_MIN_FPM   = 25.0
RUN_SPEED_MAX_FPM   = 55.0
RUN_SPEED_JITTER    = 1.5      # std-dev of per-sample jitter [fpm]

# ── Section structure ─────────────────────────────────────────────────────────
SECTION_FOOTAGE_MIN  = 800
SECTION_FOOTAGE_MAX  = 2000
IDLE_DURATION_MIN    = 60.0    # seconds (before TIME_SCALE compression)
IDLE_DURATION_MAX    = 300.0

# ── Air ramp pressure ─────────────────────────────────────────────────────────
PRESSURE_PER_FOOT   = 0.0018   # PSI/ft  →  1 000 ft ≈ 1.8 PSI
PRESSURE_NOISE_STD  = 0.008

# ── Chatter probability (progress-driven) ─────────────────────────────────────
#
#   P(burst starts this second) uses a piecewise formula dominated by normalised
#   section progress (0 = start, 1 = section cut).
#
#   The shape is a smooth cubic ramp clamped to [RANDOM_FLOOR, MAX_RATE]:
#
#       p(x) = MAX_RATE · clamp((x − ONSET) / (1 − ONSET), 0, 1)^EXPONENT
#
CHATTER_ONSET_PROGRESS = 0.30  # no chatter before 30 % of section
CHATTER_MAX_RATE       = 0.75  # probability per second at end of section
CHATTER_EXPONENT       = 2.2   # exponent of the growth curve
CHATTER_RANDOM_FLOOR   = 0.001 # tiny background rate (pressure-independent noise)

# Burst duration scales with progress: shorter early, longer late
CHATTER_DUR_MIN_EARLY  = 4.0   # seconds at progress = ONSET
CHATTER_DUR_MAX_EARLY  = 8.0
CHATTER_DUR_MIN_LATE   = 12.0  # seconds at progress = 1.0
CHATTER_DUR_MAX_LATE   = 25.0
CHATTER_MIN_GAP        = 12.0  # minimum seconds between bursts

# Chatter amplitude multiplier scales with progress
# At progress=ONSET → 1.0×CHATTER_AMPS  (subtle)
# At progress=1.0   → AMP_SCALE_PEAK×CHATTER_AMPS  (very obvious)
CHATTER_AMP_SCALE_ONSET = 0.4
CHATTER_AMP_SCALE_PEAK  = 2.5

# ── Slow process variables ────────────────────────────────────────────────────
OIL_DELIVERY_TEMP_BASE  = 252.0   # °F
OIL_RETURN_TEMP_BASE    = 251.0
TEMP_DRIFT_STD          = 0.02
TEMP_NOISE_STD          = 0.5

PT300_BASE              = 25.0
PT400_BASE              = 28.0
PRESSURE_NOISE_SLOW     = 0.03

# ─────────────────────────────── CLIENT REGISTRY ─────────────────────────────
CLIENTS: set = set()


async def client_handler(websocket, path=None):
    CLIENTS.add(websocket)
    print(f"[+] Client connected ({len(CLIENTS)} total)")
    try:
        await websocket.wait_closed()
    finally:
        CLIENTS.discard(websocket)
        print(f"[-] Client disconnected ({len(CLIENTS)} total)")


# ─────────────────────────────── HELPERS ──────────────────────────────────────

def _progress_chatter_prob(section_progress: float) -> float:
    """
    Per-second probability of a new chatter burst starting, as a function of
    normalised section progress in [0, 1].

    The curve is near-zero for the first ONSET fraction of the section, then
    rises as a power-law to CHATTER_MAX_RATE at the end of the section.
    A tiny random floor keeps the training data from being completely clean
    in the early part of every section.
    """
    shifted = max(0.0, (section_progress - CHATTER_ONSET_PROGRESS)
                  / (1.0 - CHATTER_ONSET_PROGRESS))
    driven  = CHATTER_MAX_RATE * (shifted ** CHATTER_EXPONENT)
    return min(driven + CHATTER_RANDOM_FLOOR, 0.90)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


def _chatter_amplitude_scale(section_progress: float) -> float:
    """Amplitude multiplier for CHATTER_AMPS, grows from ONSET to end of section."""
    t = max(0.0, (section_progress - CHATTER_ONSET_PROGRESS)
            / (1.0 - CHATTER_ONSET_PROGRESS))
    return _lerp(CHATTER_AMP_SCALE_ONSET, CHATTER_AMP_SCALE_PEAK, t)


def _chatter_burst_duration(section_progress: float) -> float:
    """Burst duration (seconds) scales with progress."""
    t = max(0.0, (section_progress - CHATTER_ONSET_PROGRESS)
            / (1.0 - CHATTER_ONSET_PROGRESS))
    lo = _lerp(CHATTER_DUR_MIN_EARLY, CHATTER_DUR_MIN_LATE, t)
    hi = _lerp(CHATTER_DUR_MAX_EARLY, CHATTER_DUR_MAX_LATE, t)
    return random.uniform(lo, hi)


# ─────────────────────────────── SIMULATION ──────────────────────────────────

async def simulation_loop():
    print(f"Starting simulation loop  (TIME_SCALE={TIME_SCALE:.1f}×)", flush=True)

    # ── Physical state ────────────────────────────────────────────────────────
    phases          = np.zeros(len(CHATTER_WAVELENGTHS), dtype=float)
    section_od_mean = MEAN_OD

    # ── Section state machine ─────────────────────────────────────────────────
    state               = "idle"
    section_target_ft   = 0.0
    section_footage_ft  = 0.0
    base_speed_fpm      = 0.0
    idle_end_time       = time.monotonic() + random.uniform(2.0, 4.0) / TIME_SCALE

    # ── Chatter burst state ───────────────────────────────────────────────────
    chatter_active    = False
    chatter_end_time  = 0.0
    last_chatter_end  = -CHATTER_MIN_GAP

    # ── Slow process variables ────────────────────────────────────────────────
    oil_delivery_temp = OIL_DELIVERY_TEMP_BASE + random.gauss(0, 5.0)
    oil_return_temp   = OIL_RETURN_TEMP_BASE   + random.gauss(0, 5.0)
    pt300_val         = PT300_BASE + random.gauss(0, 0.5)
    pt400_val         = PT400_BASE + random.gauss(0, 0.5)

    # ── Timing ────────────────────────────────────────────────────────────────
    last_step_time        = time.monotonic()
    accumulated_wall_dt   = 0.0

    try:
        while True:
            batch_start = time.monotonic()
            samples     = []

            for _ in range(BATCH_SIZE):
                now = time.monotonic()
                dt  = now - last_step_time
                if dt <= 0:
                    dt = dt_target
                last_step_time = now

                # ── Section state machine ─────────────────────────────────────
                section_progress = 0.0   # default for idle

                if state == "idle":
                    current_speed_fpm  = 0.0
                    air_ramp_pressure  = 0.0
                    section_footage_ft = 0.0
                    chatter_active     = False

                    if now >= idle_end_time:
                        state              = "running"
                        section_target_ft  = random.uniform(SECTION_FOOTAGE_MIN,
                                                             SECTION_FOOTAGE_MAX)
                        base_speed_fpm     = random.uniform(RUN_SPEED_MIN_FPM,
                                                             RUN_SPEED_MAX_FPM)
                        section_footage_ft = 0.0
                        section_od_mean    = MEAN_OD + random.gauss(0, OD_VARIANCE * 0.3)
                        last_chatter_end   = now - CHATTER_MIN_GAP
                        print(
                            f"[{now:.1f}] NEW SECTION — target={section_target_ft:.0f} ft, "
                            f"speed={base_speed_fpm:.1f} fpm, "
                            f"OD_mean={section_od_mean:.5f}\"",
                            flush=True,
                        )

                else:  # "running"
                    current_speed_fpm = max(
                        0.0,
                        base_speed_fpm + random.gauss(0.0, RUN_SPEED_JITTER)
                    )

                    # Footage accumulates (TIME_SCALE compresses section length)
                    section_footage_ft += (current_speed_fpm / 60.0) * dt * TIME_SCALE

                    section_progress = min(section_footage_ft / max(section_target_ft, 1.0), 1.0)

                    # Air ramp pressure: linear ramp with footage
                    air_ramp_pressure = max(
                        0.0,
                        PRESSURE_PER_FOOT * section_footage_ft
                        + random.gauss(0.0, PRESSURE_NOISE_STD)
                    )

                    if section_footage_ft >= section_target_ft:
                        state             = "idle"
                        idle_duration     = random.uniform(IDLE_DURATION_MIN,
                                                            IDLE_DURATION_MAX)
                        idle_end_time     = now + idle_duration / TIME_SCALE
                        chatter_active    = False
                        print(
                            f"[{now:.1f}] SECTION CUT — "
                            f"footage={section_footage_ft:.0f} ft, "
                            f"peak_pressure={air_ramp_pressure:.3f} PSI, "
                            f"idle={idle_duration/TIME_SCALE:.0f}s real",
                            flush=True,
                        )
                        current_speed_fpm  = 0.0
                        air_ramp_pressure  = 0.0
                        section_footage_ft = 0.0
                        section_progress   = 0.0

                # ── Chatter burst state machine (1-Hz gate) ───────────────────
                accumulated_wall_dt += dt
                if accumulated_wall_dt >= 1.0 and state == "running":
                    accumulated_wall_dt = 0.0

                    if chatter_active and now >= chatter_end_time:
                        chatter_active   = False
                        last_chatter_end = now
                        print(
                            f"  [chatter END]   progress={section_progress:.2f}  "
                            f"pressure={air_ramp_pressure:.3f} PSI",
                            flush=True,
                        )

                    if (not chatter_active
                            and (now - last_chatter_end) >= CHATTER_MIN_GAP):
                        p = _progress_chatter_prob(section_progress)
                        if random.random() < p:
                            chatter_active   = True
                            duration         = _chatter_burst_duration(section_progress)
                            chatter_end_time = now + duration
                            print(
                                f"  [chatter START] progress={section_progress:.2f}  "
                                f"pressure={air_ramp_pressure:.3f} PSI  "
                                f"p={p:.3f}  dur={duration:.1f}s  "
                                f"footage={section_footage_ft:.0f} ft",
                                flush=True,
                            )
                elif accumulated_wall_dt >= 1.0:
                    accumulated_wall_dt = 0.0

                # ── OD signal ─────────────────────────────────────────────────
                od           = section_od_mean
                speed_in_s   = current_speed_fpm * (12.0 / 60.0)  # fpm → in/s
                amp_scale    = _chatter_amplitude_scale(section_progress) if chatter_active else 1.0
                active_amps  = ([a * amp_scale for a in CHATTER_AMPS]
                                if chatter_active else NORMAL_AMPS)

                for i, (lam_in, amp_in) in enumerate(zip(CHATTER_WAVELENGTHS, active_amps)):
                    freq_hz    = (speed_in_s / lam_in) if lam_in > 0 else 0.0
                    phases[i] += 2.0 * math.pi * freq_hz * dt
                    od        += amp_in * math.sin(phases[i])

                if chatter_active:
                    noise_std = NOISE_CHATTER * amp_scale
                elif current_speed_fpm > 0.0:
                    noise_std = NOISE_RUNNING
                else:
                    noise_std = NOISE_STOPPED
                od += random.gauss(0.0, noise_std)

                # ── Ovality (elevated during chatter, scales with amplitude) ──
                base_ovality = 0.001 + abs(od - section_od_mean) * 0.3
                if chatter_active:
                    base_ovality *= random.uniform(3.0, 8.0) * amp_scale
                ovality = max(0.0002, base_ovality + random.gauss(0.0, 0.0002))

                # ── Slow process variables (random walk) ──────────────────────
                oil_delivery_temp += random.gauss(0.0, TEMP_DRIFT_STD)
                oil_delivery_temp  = float(np.clip(oil_delivery_temp,
                                                   OIL_DELIVERY_TEMP_BASE - 20,
                                                   OIL_DELIVERY_TEMP_BASE + 20))
                oil_return_temp   += random.gauss(0.0, TEMP_DRIFT_STD)
                oil_return_temp    = float(np.clip(oil_return_temp,
                                                   OIL_RETURN_TEMP_BASE - 20,
                                                   OIL_RETURN_TEMP_BASE + 20))
                pt300_val += random.gauss(0.0, PRESSURE_NOISE_SLOW)
                pt300_val  = float(np.clip(pt300_val, PT300_BASE - 3, PT300_BASE + 3))
                pt400_val += random.gauss(0.0, PRESSURE_NOISE_SLOW)
                pt400_val  = float(np.clip(pt400_val, PT400_BASE - 3, PT400_BASE + 3))

                ts_str = datetime.now(timezone.utc).isoformat()

                samples.append({
                    "t_stamp":                        ts_str,
                    # ── Primary signals ────────────────────────────────────────
                    "NDC_System_OD_Value":            float(od),
                    "NDC_System_Ovality_Value":       float(ovality),
                    "YS_Pullout1_Act_Speed_fpm":      float(current_speed_fpm),
                    # ── Section / air ramp ─────────────────────────────────────
                    "AirRampPressure_Val":             float(air_ramp_pressure),
                    "FtCounters_AirRampFootage_Total": float(section_footage_ft),
                    # ── Derived: normalised section progress (0 → 1) ───────────
                    # Included so the training page can use it as a context feature.
                    # In production this can be recomputed from footage / max-footage.
                    "SectionProgress_Normalized":     float(section_progress),
                    # ── Slow process variables ─────────────────────────────────
                    "OilHeater_DeliveryTemp_F":       float(
                        oil_delivery_temp + random.gauss(0.0, TEMP_NOISE_STD)),
                    "OilHeater_ReturnTemp_F":         float(
                        oil_return_temp   + random.gauss(0.0, TEMP_NOISE_STD)),
                    "PTs_PT_300_Val":                 float(pt300_val),
                    "PTs_PT_400_Val":                 float(pt400_val),
                    # ── Status flags ───────────────────────────────────────────
                    "LineStatus_Running":             int(state == "running"),
                    # ── Ground-truth labels (training / eval only) ─────────────
                    "_chatter_active":                int(chatter_active),
                    "_section_progress":             float(section_progress),
                })

            # ── Broadcast ─────────────────────────────────────────────────────
            if CLIENTS:
                msg  = json.dumps({"samples": samples})
                dead = set()
                for ws in list(CLIENTS):
                    try:
                        await ws.send(msg)
                    except websockets.exceptions.ConnectionClosed:
                        dead.add(ws)
                CLIENTS.difference_update(dead)

            # ── Sleep to maintain 2400 Hz OD rate ─────────────────────────────
            batch_duration = BATCH_SIZE * dt_target
            sleep_time     = max(0.0, batch_duration - (time.monotonic() - batch_start))
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    except asyncio.CancelledError:
        print("Simulation cancelled.")


# ─────────────────────────────── ENTRY POINT ─────────────────────────────────

async def main():
    host, port = "localhost", 6467
    server = await websockets.serve(client_handler, host, port)
    print(f"WebSocket server → ws://{host}:{port}  (TIME_SCALE={TIME_SCALE}×)")
    print()
    print("Signal key  →  chatter relationship:")
    print(f"  SectionProgress_Normalized    0→1 within each section")
    print(f"  AirRampPressure_Val           ramps with footage ({PRESSURE_PER_FOOT} PSI/ft)")
    print(f"  _chatter_active               ground-truth label")
    print()
    print("Chatter onset curve:")
    for pct in [0, 25, 35, 45, 55, 65, 75, 85, 95, 100]:
        p = _progress_chatter_prob(pct / 100)
        bar = "█" * int(p * 30)
        print(f"  progress={pct:3d}%   p={p:.3f}  {bar}")
    print()

    sim_task = asyncio.create_task(simulation_loop())
    try:
        await asyncio.Future()
    finally:
        sim_task.cancel()
        await sim_task
        server.close()
        await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())