# %%
import numpy as np

from egifa.data import get_voiced_meta
from utils.audio import frame_signal


def get_voiced_runs(
    path_contains=None,
    frame_len_msec=32.0,
    hop_ms=16.0,
    num_vi_restarts=1,
    dtype=np.float64,
    **smooth_dgf_kwargs,
):
    for v in get_voiced_meta(path_contains=path_contains, **smooth_dgf_kwargs):
        t = v["smooth"]["t_samples"].astype(dtype)
        x = v["smooth"]["speech"].astype(dtype)
        u = v["smooth"]["gf"].astype(dtype)
        du = v["smooth"]["dgf"].astype(dtype)
        tau = v["smooth"]["tau"].astype(dtype)
        assert len(x) == len(u) == len(du) == len(t) == len(tau)

        fs = v["smooth"]["fs"]
        frame_len = int(frame_len_msec / 1000 * fs)
        hop = int(hop_ms / 1000 * fs)

        if len(t) < frame_len:  # ditch voiced groups shorter than one frame
            continue

        t_frames = frame_signal(t, frame_len, hop)
        x_frames = frame_signal(x, frame_len, hop)
        u_frames = frame_signal(u, frame_len, hop)
        du_frames = frame_signal(du, frame_len, hop)
        tau_frames = frame_signal(tau, frame_len, hop)

        for frame_index, (t, x, u, du, tau) in enumerate(
            zip(t_frames, x_frames, u_frames, du_frames, tau_frames)
        ):
            t_min, t_max = t[0], t[-1]
            loc = np.where((t_min <= v["gci"]) & (v["gci"] <= t_max))[0]

            gci = v["gci"][loc]
            goi = v["goi"][loc]
            oq = v["oq"][loc[:-1]]
            periods_ms = v["periods_ms"][loc[:-1]]

            for restart_index in range(num_vi_restarts):
                f = {
                    "fs": fs,
                    "t_samples": t,
                    "tau": tau,
                    "speech": x,
                    "gf": u,
                    "dgf": du,
                    "gci": gci,
                    "goi": goi,
                    "oq": oq,
                    "periods_ms": periods_ms,
                    "frame_index": frame_index,
                    "restart_index": restart_index,
                }

                yield {"group": v, "frame": f}


if __name__ == "__main__":
    runs = list(get_voiced_runs())
    print("Total runs:", len(runs))

    x = np.vstack([r["frame"]["speech"] for r in runs])
    print("Data shape:", x.shape)
