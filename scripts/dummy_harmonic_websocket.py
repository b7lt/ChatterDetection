import asyncio
import json
import random
import time
from datetime import datetime, timezone

import numpy as np
import websockets

# ---------------- CONFIG ----------------
fs = 2400.0                  # websocket sample rate [Hz]
dt_target = 1.0 / fs        # target time step [s]

mean_od = 0.5           # target OD [inches]
drift_per_ft = 3.5e-05     # slow drift per foot

# Chatter patterns (spatial) along the tube
chatter_wavelengths = [1.0, 2.0] # [inches]
base_amp = 0.002 * (mean_od / 0.50)
chatter_amps = [0.6 * base_amp, 0.7 * base_amp] # [inches]

noise_std_running = 0.0002 * (mean_od / 0.50)   # noise when line is moving [inches]
noise_std_stopped = 0.00002 * (mean_od / 0.50)  # noise when line is stopped [inches]

# Random segment behavior (real time)
MIN_SEG_DURATION = 5.0      # [s]
MAX_SEG_DURATION = 20.0     # [s]

RUN_SPEED_MIN_FPM = 120.0
RUN_SPEED_MAX_FPM = 220.0
RUN_SPEED_JITTER_STD = 3.0  # small jitter around the "constant" speed


def sample_segment_duration():
    """Random segment length in seconds."""
    return random.uniform(MIN_SEG_DURATION, MAX_SEG_DURATION)


def sample_run_speed():
    """Random near-constant running speed in fpm."""
    return random.uniform(RUN_SPEED_MIN_FPM, RUN_SPEED_MAX_FPM)


# ---------------- GLOBAL CLIENT REGISTRY ----------------
CLIENTS = set()


async def client_handler(websocket, path=None):
    """
    Per-client handler: just register/unregister the client.
    The simulation loop is global and broadcasts to all CLIENTS.
    """
    CLIENTS.add(websocket)
    print(f"Client connected. Now {len(CLIENTS)} client(s) connected.")
    try:
        # We don't expect incoming messages; just wait until the client disconnects.
        await websocket.wait_closed()
    finally:
        CLIENTS.discard(websocket)
        print(f"Client disconnected. Now {len(CLIENTS)} client(s) connected.")


async def simulation_loop():
    """
    Single global simulation producing OD + speed and broadcasting to all clients.
    All clients see the same values at the same time.
    """
    print("Starting global simulation loop...")

    # Real-time clock
    now = time.monotonic()
    last_step_time = now

    # Segment state
    state = "stopped"  # or "running"
    base_speed_fpm = 0.0
    next_segment_switch = now + sample_segment_duration()

    # Physical state
    length_in = 0.0  # cumulative length [in]
    phases = np.zeros(len(chatter_wavelengths), dtype=float)

    try:
        while True:
            now = time.monotonic()
            dt = now - last_step_time
            if dt <= 0:
                dt = dt_target
            last_step_time = now

            # --- Segment switching based on wall-clock time ---
            if now >= next_segment_switch:
                if state == "stopped":
                    state = "running"
                    base_speed_fpm = sample_run_speed()
                # else:
                #     state = "stopped"
                #     base_speed_fpm = 0.0

                seg_dur = sample_segment_duration()
                next_segment_switch = now + seg_dur

                print(
                    f"[{now:.3f}] New segment: state={state}, "
                    f"base_speed_fpm={base_speed_fpm:.1f}, duration={seg_dur:.1f}s"
                )

            # --- Current speed (with small jitter when running) ---
            if state == "running":
                current_speed_fpm = base_speed_fpm + random.gauss(0.0, RUN_SPEED_JITTER_STD)
                current_speed_fpm = max(current_speed_fpm, 0.0)
            else:
                current_speed_fpm = 0.0

            # Convert speed to inches per second
            speed_in_s = current_speed_fpm * (12.0 / 60.0)  # 1 fpm = 0.2 in/s

            # Update length along the tube based on actual dt
            length_in += speed_in_s * dt
            length_ft = length_in / 12.0

            # Base OD = mean + drift
            od = mean_od + drift_per_ft * length_ft

            # Add chatter components via phase accumulation
            for i, (lam_in, amp_in) in enumerate(
                zip(chatter_wavelengths, chatter_amps)
            ):
                if lam_in > 0:
                    freq_hz = speed_in_s / lam_in  # f = v / λ
                else:
                    freq_hz = 0.0
                phases[i] += 2.0 * np.pi * freq_hz * dt
                od += amp_in * np.sin(phases[i])

            # Add noise, lower when stopped
            noise_std = noise_std_running if current_speed_fpm > 0.0 else noise_std_stopped
            od += random.gauss(0.0, noise_std)

            # Real-time timestamp
            current_time = datetime.now(timezone.utc)
            ts_str = current_time.isoformat()

            msg = {
                "t_stamp": ts_str,
                "od": float(od),
                "speed": float(current_speed_fpm),
            }

            # Broadcast to all connected clients
            if CLIENTS:
                # Make a snapshot so we don't break if CLIENTS changes mid-loop
                to_send = list(CLIENTS)
                msg_text = json.dumps(msg)
                # Send to all, dropping any that error out
                for ws in to_send:
                    try:
                        await ws.send(msg_text)
                    except websockets.exceptions.ConnectionClosed:
                        CLIENTS.discard(ws)

            # Sleep to maintain ~fs Hz
            sleep_time = max(0.0, dt_target - (time.monotonic() - now))
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    except asyncio.CancelledError:
        print("Simulation loop cancelled; shutting down.")


async def main():
    host = "localhost"
    port = 6467

    # Start server
    server = await websockets.serve(client_handler, host, port)
    print(f"WebSocket server running at ws://{host}:{port}")

    # Start the single global simulation task
    sim_task = asyncio.create_task(simulation_loop())

    try:
        await asyncio.Future()  # run forever
    finally:
        sim_task.cancel()
        await sim_task
        server.close()
        await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
