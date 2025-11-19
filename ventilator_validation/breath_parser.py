import numpy as np
from scipy.signal import find_peaks

def segment_breaths(time, flow, pressure, co2, fs):
    """
    Segments breaths using Flow (if available/high-freq) or Pressure/CO2 minima.
    
    Args:
        time, flow, pressure, co2: Signal arrays
        fs: Sampling frequency
    """
    breaths = []
    
    # Strategy:
    # 1. Try Flow Zero-Crossing (if flow looks like a waveform)
    # 2. If Flow is flat/numeric, use Pressure Minima (PEEP)
    # 3. Or CO2 waveform (Capnography) - start of expiration is rise in CO2, start of inspiration is drop to 0.
    
    # Check if flow is high-fidelity
    # Calculate variance or check if it changes frequently
    is_flow_waveform = np.std(np.diff(flow)) > 0.1 # Heuristic
    
    if is_flow_waveform:
        # Use Zero-Crossing on Flow (Inspiration starts when flow goes positive)
        # ... (Same logic as before)
        insp_starts = np.where((flow[:-1] <= 0) & (flow[1:] > 0))[0]
    else:
        # Use Pressure Minima (End of Expiration / Start of Inspiration)
        # Pressure usually drops to PEEP before next breath
        # We can find local minima in Pressure
        # Invert pressure to find peaks
        # This is tricky because pressure is plateau-ish during insp.
        # Better to use CO2 if available.
        
        # CO2 (Capnogram):
        # Inspiration: CO2 is near 0 (fresh gas)
        # Expiration: CO2 rises (ETCO2)
        # Transition from High CO2 to Low CO2 = Start of Inspiration
        
        # Let's find where CO2 drops below a threshold (e.g. 5 mmHg)
        # Assuming CO2 is in mmHg.
        # CO2 signal is usually square-wave like.
        
        # Find indices where CO2 crosses down threshold
        threshold = 5.0
        insp_starts = np.where((co2[:-1] > threshold) & (co2[1:] <= threshold))[0]
        
        if len(insp_starts) < 2:
            # Fallback to Pressure Minima
            # Find valleys in pressure
            # We use scipy.signal.find_peaks on inverted pressure
            peaks, _ = find_peaks(-pressure, distance=int(2*fs)) # Assume at least 2s between breaths
            insp_starts = peaks

    if len(insp_starts) < 2:
        return breaths
        
    for i in range(len(insp_starts) - 1):
        start_idx = insp_starts[i]
        end_idx = insp_starts[i+1]
        
        b_time = time[start_idx:end_idx]
        b_flow = flow[start_idx:end_idx]
        b_pressure = pressure[start_idx:end_idx]
        b_co2 = co2[start_idx:end_idx]
        
        duration = len(b_time) / fs
        
        if duration < 0.5 or duration > 10.0:
            continue
            
        breaths.append({
            'time': b_time,
            'flow': b_flow,
            'pressure': b_pressure,
            'co2': b_co2,
            'duration': duration
        })
        
    return breaths
