"""
dummy_harmonic_websocket.py
===========================
Simulates the tubing manufacturing process at 2400 Hz, including:
  - Batch/section structure: line runs until target footage, then goes idle
  - Air ramp pressure that rises linearly with footage (~0.0018 PSI/ft)
  - Chatter probability that increases once pressure exceeds a threshold
  - Realistic OD (~0.069"), speed (25–55 fpm), ovality, oil temps, pressures
    derived from real historian data (tagHistoryData_04-01-00_04-08-10.xlsx)

TIME_SCALE compresses section time for faster testing:
  TIME_SCALE = 1.0  → real time  (~20–50 min per section)
  TIME_SCALE = 10.0 → 10× faster (~2–5 min per section)  ← default
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

TIME_SCALE = 10.0        # compress section progression; does NOT alter 2400 Hz OD rate

fs          = 2400.0     # OD sensor sample rate [Hz]
dt_target   = 1.0 / fs
BATCH_SIZE  = 40         # samples per WebSocket message → ~60 msg/s

# ── OD / geometry (from real data) ───────────────────────────────────────────
MEAN_OD     = 0.069      # inches
OD_VARIANCE = 0.005      # ±inches natural variation across sections

# Chatter harmonics (spatial)
CHATTER_WAVELENGTHS = [1.0, 2.0]          # inches (spatial period on tube)
base_amp             = 0.002 * (MEAN_OD / 0.069)
NORMAL_AMPS  = [0.6 * base_amp, 0.7 * base_amp]   # quiet running
CHATTER_AMPS = [8.0 * base_amp, 6.0 * base_amp]   # chatter burst

NOISE_RUNNING = 0.0002 * (MEAN_OD / 0.069)
NOISE_STOPPED = 0.00002 * (MEAN_OD / 0.069)
NOISE_CHATTER = 0.003   * (MEAN_OD / 0.069)

# ── Speed (from real data: 25–55 fpm) ────────────────────────────────────────
RUN_SPEED_MIN_FPM   = 25.0
RUN_SPEED_MAX_FPM   = 55.0
RUN_SPEED_JITTER    = 1.5      # std-dev of per-sample jitter [fpm]

# ── Section structure (from real data) ───────────────────────────────────────
SECTION_FOOTAGE_MIN  = 800     # ft
SECTION_FOOTAGE_MAX  = 2000    # ft
IDLE_DURATION_MIN    = 60.0    # seconds (real: 300–1200 s; compressed by TIME_SCALE)
IDLE_DURATION_MAX    = 300.0   # seconds

# ── Air ramp pressure (from real data fit) ───────────────────────────────────
PRESSURE_PER_FOOT   = 0.0018   # PSI/ft  →  1000 ft ≈ 1.8 PSI, 1500 ft ≈ 2.7 PSI
PRESSURE_NOISE_STD  = 0.008    # PSI — small per-sample noise on top of ramp

# ── Chatter probability (sigmoid gate on pressure) ───────────────────────────
# P(chatter start this second) = sigmoid((pressure - THRESH) * SCALE) * BASE_RATE
CHATTER_PRESSURE_THRESHOLD = 1.5    # PSI — below this, chatter almost never starts
CHATTER_PRESSURE_SCALE     = 3.0    # steepness of sigmoid
CHATTER_BASE_RATE          = 0.015  # max fraction of seconds that start a burst
CHATTER_DURATION_MIN       = 8.0    # seconds
CHATTER_DURATION_MAX       = 18.0   # seconds
CHATTER_MIN_GAP            = 15.0   # minimum seconds between bursts

# Random low-pressure chatter (false-positive noise in training data)
CHATTER_RANDOM_BASE_RATE   = 0.0008  # ~0.05 bursts/minute regardless of pressure

# ── Slow process variables (from real data ranges) ────────────────────────────
OIL_DELIVERY_TEMP_BASE  = 252.0     # °F
OIL_RETURN_TEMP_BASE    = 251.0     # °F
TEMP_DRIFT_STD          = 0.02      # °F per sample (slow random walk)
TEMP_NOISE_STD          = 0.5       # °F measurement noise

PT300_BASE              = 25.0      # PSI (main line pressure)
PT400_BASE              = 28.0      # PSI
PRESSURE_NOISE_SLOW     = 0.03      # PSI per-sample drift

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


# ─────────────────────────────── SIMULATION ──────────────────────────────────

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _chatter_start_prob(pressure: float) -> float:
    """Per-second probability of a chatter burst starting at this pressure."""
    pressure_driven = (
        _sigmoid((pressure - CHATTER_PRESSURE_THRESHOLD) * CHATTER_PRESSURE_SCALE)
        * CHATTER_BASE_RATE
    )
    random_component = CHATTER_RANDOM_BASE_RATE
    return pressure_driven + random_component


async def simulation_loop():
    print("Starting simulation loop  (TIME_SCALE=%.1f×)" % TIME_SCALE, flush=True)

    # ── Physical state ────────────────────────────────────────────────────────
    phases         = np.zeros(len(CHATTER_WAVELENGTHS), dtype=float)
    section_od_mean = MEAN_OD  # may vary slightly per section

    # ── Section state machine ─────────────────────────────────────────────────
    #   States: "idle" | "running"
    state              = "idle"
    section_target_ft  = 0.0
    section_footage_ft = 0.0     # footage within current section (resets each cut)
    base_speed_fpm     = 0.0
    idle_end_time      = time.monotonic() + random.uniform(2.0, 4.0) / TIME_SCALE  # short first idle

    # ── Chatter burst state ───────────────────────────────────────────────────
    chatter_active   = False
    chatter_end_time = 0.0
    last_chatter_end = -CHATTER_MIN_GAP  # allow burst immediately if conditions met

    # ── Slow process variables (random-walk state) ────────────────────────────
    oil_delivery_temp = OIL_DELIVERY_TEMP_BASE + random.gauss(0, 5.0)
    oil_return_temp   = OIL_RETURN_TEMP_BASE   + random.gauss(0, 5.0)
    pt300_val         = PT300_BASE + random.gauss(0, 0.5)
    pt400_val         = PT400_BASE + random.gauss(0, 0.5)

    # ── Timing ────────────────────────────────────────────────────────────────
    last_step_time   = time.monotonic()
    last_chatter_check_sec = -1.0   # wall-clock second of last chatter-start check

    # Track accumulated wall-time for chatter check (1-Hz gate)
    accumulated_wall_dt = 0.0

    try:
        while True:
            batch_start = time.monotonic()
            samples = []

            for _ in range(BATCH_SIZE):
                now = time.monotonic()
                dt  = now - last_step_time
                if dt <= 0:
                    dt = dt_target
                last_step_time = now

                # ── Section state machine ─────────────────────────────────────
                if state == "idle":
                    current_speed_fpm     = 0.0
                    air_ramp_pressure     = 0.0
                    section_footage_ft    = 0.0
                    chatter_active        = False

                    if now >= idle_end_time:
                        # Start a new section
                        state             = "running"
                        section_target_ft = random.uniform(SECTION_FOOTAGE_MIN,
                                                            SECTION_FOOTAGE_MAX)
                        base_speed_fpm    = random.uniform(RUN_SPEED_MIN_FPM,
                                                            RUN_SPEED_MAX_FPM)
                        section_footage_ft = 0.0
                        section_od_mean   = MEAN_OD + random.gauss(0, OD_VARIANCE * 0.3)
                        last_chatter_end  = now - CHATTER_MIN_GAP  # fresh gap timer
                        print(
                            f"[{now:.1f}] NEW SECTION — target={section_target_ft:.0f} ft, "
                            f"speed={base_speed_fpm:.1f} fpm, "
                            f"OD_mean={section_od_mean:.4f}\"",
                            flush=True,
                        )

                else:  # state == "running"
                    # Speed with small jitter
                    current_speed_fpm = (base_speed_fpm
                                         + random.gauss(0.0, RUN_SPEED_JITTER))
                    current_speed_fpm = max(current_speed_fpm, 0.0)

                    # Footage accumulates (TIME_SCALE compresses section length)
                    speed_in_s = current_speed_fpm * (12.0 / 60.0)   # fpm → in/s
                    section_footage_ft += (current_speed_fpm / 60.0) * dt * TIME_SCALE

                    # Air ramp pressure: linear ramp with footage
                    air_ramp_pressure = (PRESSURE_PER_FOOT * section_footage_ft
                                         + random.gauss(0.0, PRESSURE_NOISE_STD))
                    air_ramp_pressure = max(air_ramp_pressure, 0.0)

                    # Check for section end
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
                            f"idle={idle_duration/TIME_SCALE:.0f}s (real={idle_duration:.0f}s)",
                            flush=True,
                        )
                        # Snap values to idle state for this sample
                        current_speed_fpm  = 0.0
                        air_ramp_pressure  = 0.0
                        section_footage_ft = 0.0

                # ── Chatter burst state machine (checked once per wall-clock second) ─
                accumulated_wall_dt += dt
                if accumulated_wall_dt >= 1.0 and state == "running":
                    accumulated_wall_dt = 0.0
                    # End an active burst?
                    if chatter_active and now >= chatter_end_time:
                        chatter_active   = False
                        last_chatter_end = now
                        print(f"  [chatter END] pressure={air_ramp_pressure:.3f} PSI", flush=True)
                    # Start a new burst?
                    if (not chatter_active
                            and (now - last_chatter_end) >= CHATTER_MIN_GAP):
                        p = _chatter_start_prob(air_ramp_pressure)
                        if random.random() < p:
                            chatter_active   = True
                            duration         = random.uniform(CHATTER_DURATION_MIN,
                                                               CHATTER_DURATION_MAX)
                            chatter_end_time = now + duration
                            print(
                                f"  [chatter START] pressure={air_ramp_pressure:.3f} PSI "
                                f"(p={p:.4f}), duration={duration:.1f}s, "
                                f"footage={section_footage_ft:.0f} ft",
                                flush=True,
                            )
                elif accumulated_wall_dt >= 1.0:
                    accumulated_wall_dt = 0.0  # drain counter while idle

                # ── OD signal ─────────────────────────────────────────────────
                od = section_od_mean
                speed_in_s = current_speed_fpm * (12.0 / 60.0)
                active_amps = CHATTER_AMPS if chatter_active else NORMAL_AMPS
                for i, (lam_in, amp_in) in enumerate(
                    zip(CHATTER_WAVELENGTHS, active_amps)
                ):
                    freq_hz   = (speed_in_s / lam_in) if lam_in > 0 else 0.0
                    phases[i] += 2.0 * math.pi * freq_hz * dt
                    od        += amp_in * math.sin(phases[i])

                if chatter_active:
                    noise_std = NOISE_CHATTER
                elif current_speed_fpm > 0.0:
                    noise_std = NOISE_RUNNING
                else:
                    noise_std = NOISE_STOPPED
                od += random.gauss(0.0, noise_std)

                # ── Ovality (correlated with OD variance; elevated during chatter) ─
                base_ovality = 0.001 + abs(od - section_od_mean) * 0.3
                if chatter_active:
                    base_ovality *= random.uniform(3.0, 8.0)
                ovality = max(0.0002, base_ovality + random.gauss(0.0, 0.0002))

                # ── Slow process variables (random walk) ──────────────────────
                oil_delivery_temp += random.gauss(0.0, TEMP_DRIFT_STD)
                oil_delivery_temp  = np.clip(oil_delivery_temp,
                                             OIL_DELIVERY_TEMP_BASE - 20,
                                             OIL_DELIVERY_TEMP_BASE + 20)
                oil_return_temp   += random.gauss(0.0, TEMP_DRIFT_STD)
                oil_return_temp    = np.clip(oil_return_temp,
                                             OIL_RETURN_TEMP_BASE - 20,
                                             OIL_RETURN_TEMP_BASE + 20)
                pt300_val += random.gauss(0.0, PRESSURE_NOISE_SLOW)
                pt300_val  = np.clip(pt300_val, PT300_BASE - 3, PT300_BASE + 3)
                pt400_val += random.gauss(0.0, PRESSURE_NOISE_SLOW)
                pt400_val  = np.clip(pt400_val, PT400_BASE - 3, PT400_BASE + 3)

                ts_str = datetime.now(timezone.utc).isoformat()

                samples.append({
                    "t_stamp":                        ts_str,
                    # ── Primary signals ────────────────────────────────
                    "NDC_System_OD_Value":            float(od),
                    "NDC_System_Ovality_Value":       float(ovality),
                    "YS_Pullout1_Act_Speed_fpm":      float(current_speed_fpm),
                    # ── Section / air ramp ─────────────────────────────
                    "AirRampPressure_Val":             float(air_ramp_pressure),
                    "FtCounters_AirRampFootage_Total": float(section_footage_ft),
                    # ── Slow process variables ─────────────────────────
                    "OilHeater_DeliveryTemp_F":        float(
                        oil_delivery_temp + random.gauss(0.0, TEMP_NOISE_STD)),
                    "OilHeater_ReturnTemp_F":          float(
                        oil_return_temp   + random.gauss(0.0, TEMP_NOISE_STD)),
                    "PTs_PT_300_Val":                  float(pt300_val),
                    "PTs_PT_400_Val":                  float(pt400_val),
                    # ── Status flags ───────────────────────────────────
                    "LineStatus_Running":              int(state == "running"),
                    # ── Ground-truth label (for training / eval only) ──
                    "_chatter_active":                 int(chatter_active),
                })

            # ── Broadcast ─────────────────────────────────────────────────────
            if CLIENTS:
                msg = json.dumps({"samples": samples})
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
    print("Signals emitted per sample:")
    print("  NDC_System_OD_Value            — outer diameter [in]")
    print("  NDC_System_Ovality_Value       — ovality [in]")
    print("  YS_Pullout1_Act_Speed_fpm      — pullout speed [fpm]")
    print("  AirRampPressure_Val            — air ramp pressure [PSI]  ← ramps with footage")
    print("  FtCounters_AirRampFootage_Total— footage in current section [ft]")
    print("  OilHeater_DeliveryTemp_F       — oil delivery temperature [°F]")
    print("  OilHeater_ReturnTemp_F         — oil return temperature [°F]")
    print("  PTs_PT_300_Val / PTs_PT_400_Val— process pressures [PSI]")
    print("  LineStatus_Running             — 1 while running, 0 while idle")
    print("  _chatter_active                — ground-truth label (1=chatter)")
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