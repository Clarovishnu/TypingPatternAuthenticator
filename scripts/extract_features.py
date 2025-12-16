import os, json, csv, statistics
import numpy as np

RAW_DIR = 'data/raw_logs'
OUT_FILE = 'data/features.csv'


# --------------------------------------------------------------------
#  Function 1: Batch feature extraction (used during training)
# --------------------------------------------------------------------
def extract_features_from_events(events):
    """Compute dwell and flight times from one sample's key events."""
    # pair up keydowns and keyups
    pairs = []
    down_times = {}
    for ev in events:
        key, t, typ = ev['key'], ev['t'], ev['type']
        if typ == 'down':
            down_times.setdefault(key, []).append(t)
        elif typ == 'up':
            if key in down_times and down_times[key]:
                d = down_times[key].pop(0)
                pairs.append({'key': key, 'down': d, 'up': t})

    pairs.sort(key=lambda x: x['down'])

    # dwell times
    dwell = [p['up'] - p['down'] for p in pairs if p['up'] > p['down']]
    # flight times (time between one key up and next key down)
    flight = [pairs[i + 1]['down'] - pairs[i]['up'] for i in range(len(pairs) - 1)]

    def stats(arr):
        if not arr:
            return (0, 0, 0, 0)
        return (
            statistics.mean(arr),
            statistics.pstdev(arr) if len(arr) > 1 else 0,
            min(arr),
            max(arr)
        )

    d_mean, d_std, d_min, d_max = stats(dwell)
    f_mean, f_std, f_min, f_max = stats(flight)

    return {
        'dwell_mean': d_mean, 'dwell_std': d_std,
        'dwell_min': d_min, 'dwell_max': d_max,
        'flight_mean': f_mean, 'flight_std': f_std,
        'flight_min': f_min, 'flight_max': f_max,
        'n_keys': len(pairs)
    }


# --------------------------------------------------------------------
#  Function 2: Real-time feature extraction (used during prediction)
# --------------------------------------------------------------------
def extract_features_from_log(log_data):
    """
    Extract typing pattern features directly from one log JSON (for prediction).
    Compatible with the real-time /api/predict route.
    """
    events = log_data.get('events') or log_data.get('key_events')
    if not events:
        raise ValueError("No key events found in log data.")

    pairs = []
    down_times = {}
    for ev in events:
        key = ev.get('key')
        t = ev.get('t') or ev.get('time') or ev.get('press_time')
        typ = ev.get('type')
        if typ == 'down':
            down_times.setdefault(key, []).append(t)
        elif typ == 'up' and key in down_times and down_times[key]:
            d = down_times[key].pop(0)
            pairs.append({'key': key, 'down': d, 'up': t})

    pairs.sort(key=lambda x: x['down'])

    dwell = [p['up'] - p['down'] for p in pairs if p['up'] > p['down']]
    flight = [pairs[i + 1]['down'] - pairs[i]['up'] for i in range(len(pairs) - 1)]

    def stats(arr):
        if not arr:
            return (0, 0, 0, 0)
        return (
            float(np.mean(arr)),
            float(np.std(arr)),
            float(np.min(arr)),
            float(np.max(arr))
        )

    d_mean, d_std, d_min, d_max = stats(dwell)
    f_mean, f_std, f_min, f_max = stats(flight)

    return {
        'dwell_mean': d_mean, 'dwell_std': d_std,
        'dwell_min': d_min, 'dwell_max': d_max,
        'flight_mean': f_mean, 'flight_std': f_std,
        'flight_min': f_min, 'flight_max': f_max,
        'n_keys': len(pairs)
    }


# --------------------------------------------------------------------
#  Function 3: Main (only runs when executed directly)
# --------------------------------------------------------------------
def main():
    rows = []
    files = [f for f in os.listdir(RAW_DIR) if f.endswith('.json')]
    for fname in files:
        path = os.path.join(RAW_DIR, fname)
        with open(path) as f:
            payload = json.load(f)
        feats = extract_features_from_events(payload['events'])
        feats['user_id'] = payload.get('user_id')
        feats['file'] = fname
        rows.append(feats)

    # save to CSV
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f" Extracted {len(rows)} samples to {OUT_FILE}")


# --------------------------------------------------------------------
#  Script Entry Point
# --------------------------------------------------------------------
if __name__ == '__main__':
    main()
