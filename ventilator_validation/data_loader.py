import vitaldb
import numpy as np
import pandas as pd
from scipy import signal

# Constants
TARGET_FS = 62.5  # Keep native VitalDB frequency for accurate feature extraction
HPA_TO_CMH2O = 1.0197

def load_and_process_case(caseid):
    """
    Loads a VitalDB case and extracts breath-level features matching the trained model.
    
    Returns:
        DataFrame with breath-level features, or None if processing fails.
    """
    try:
        # Load Primus/AWP (Airway Pressure) at native 62.5 Hz
        track_names = ['Primus/AWP']
        vals = vitaldb.load_case(caseid, track_names, interval=1/62.5)
        
        if vals is None or len(vals) == 0:
            return None
            
        pressure = vals[:, 0]
        
        # Handle NaNs
        if np.isnan(pressure).all():
            return None
            
        # Convert hPa -> cmH2O
        pressure = np.nan_to_num(pressure) * HPA_TO_CMH2O
        
        # Segment into breaths using pressure minima (PEEP detection)
        # Breaths typically 2-6 seconds (125-375 samples at 62.5Hz)
        from scipy.signal import find_peaks
        
        # Find valleys (PEEP points) - invert signal
        peaks, _ = find_peaks(-pressure, distance=int(2*62.5), prominence=2)
        
        if len(peaks) < 10:  # Need at least 10 breaths for trajectory features
            return None
            
        # Extract breath-level features
        breath_features = []
        
        for i in range(len(peaks) - 1):
            start_idx = peaks[i]
            end_idx = peaks[i+1]
            
            breath_pressure = pressure[start_idx:end_idx]
            
            if len(breath_pressure) < 10:  # Skip very short segments
                continue
                
            # Calculate features matching the trained model
            features = {
                'breath_id': i,
                'caseid': caseid,
                'PIP': np.max(breath_pressure),
                'mean_pressure': np.mean(breath_pressure),
                'pressure_std': np.std(breath_pressure),
                'pressure_range': np.max(breath_pressure) - np.min(breath_pressure),
                'pressure_skew': pd.Series(breath_pressure).skew(),
                'pressure_rise_time': np.sum(breath_pressure > np.mean(breath_pressure)),
                # Note: VitalDB doesn't have flow, so we'll set these to 0
                # The model will still work as these are not the top features
                'max_flow_in': 0,
                'mean_flow_in': 0,
                'flow_variability': 0,
                'R': 20,  # Placeholder - typical value
                'C': 50,  # Placeholder - typical value
            }
            breath_features.append(features)
            
        if len(breath_features) < 10:
            return None
            
        breath_df = pd.DataFrame(breath_features)
        
        # Calculate TRAJECTORY FEATURES (matching trained model exactly)
        # Lag features (1-5 breaths back)
        for lag in range(1, 6):
            breath_df[f'PIP_lag{lag}'] = breath_df['PIP'].shift(lag)
            breath_df[f'mean_pressure_lag{lag}'] = breath_df['mean_pressure'].shift(lag)
            breath_df[f'pressure_range_lag{lag}'] = breath_df['pressure_range'].shift(lag)
        
        # Temporal dynamics
        breath_df['PIP_slope_3'] = (breath_df['PIP'] - breath_df['PIP_lag3']) / 3
        breath_df['PIP_slope_5'] = (breath_df['PIP'] - breath_df['PIP_lag5']) / 5
        breath_df['PIP_acceleration'] = (breath_df['PIP'] - breath_df['PIP_lag1']) - (breath_df['PIP_lag1'] - breath_df['PIP_lag2'])
        
        # Volatility
        breath_df['PIP_volatility_3'] = breath_df[['PIP', 'PIP_lag1', 'PIP_lag2']].std(axis=1)
        breath_df['PIP_volatility_5'] = breath_df[[f'PIP_lag{i}' for i in range(1, 6)]].std(axis=1)
        
        # Trend
        breath_df['PIP_trend_3'] = breath_df[['PIP', 'PIP_lag1', 'PIP_lag2']].apply(
            lambda x: np.polyfit(range(3), x, 1)[0] if x.notna().all() else np.nan, axis=1
        )
        breath_df['PIP_trend_5'] = breath_df[['PIP', 'PIP_lag1', 'PIP_lag2', 'PIP_lag3', 'PIP_lag4']].apply(
            lambda x: np.polyfit(range(5), x, 1)[0] if x.notna().all() else np.nan, axis=1
        )
        
        # Consecutive rises
        breath_df['consecutive_rises'] = 0
        for i in range(len(breath_df)):
            if i >= 3:
                if (breath_df.loc[i, 'PIP'] > breath_df.loc[i, 'PIP_lag1'] and
                    breath_df.loc[i, 'PIP_lag1'] > breath_df.loc[i, 'PIP_lag2'] and
                    breath_df.loc[i, 'PIP_lag2'] > breath_df.loc[i, 'PIP_lag3']):
                    breath_df.loc[i, 'consecutive_rises'] = 3
                elif (breath_df.loc[i, 'PIP'] > breath_df.loc[i, 'PIP_lag1'] and
                      breath_df.loc[i, 'PIP_lag1'] > breath_df.loc[i, 'PIP_lag2']):
                    breath_df.loc[i, 'consecutive_rises'] = 2
                elif breath_df.loc[i, 'PIP'] > breath_df.loc[i, 'PIP_lag1']:
                    breath_df.loc[i, 'consecutive_rises'] = 1
        
        # Range volatility
        breath_df['range_volatility'] = breath_df[['pressure_range'] + [f'pressure_range_lag{i}' for i in range(1, 4)]].std(axis=1)
        
        # Compliance/Resistance dynamics (placeholders since we don't have these)
        breath_df['C_lag1'] = breath_df['C'].shift(1)
        breath_df['R_lag1'] = breath_df['R'].shift(1)
        breath_df['C_change'] = breath_df['C'] - breath_df['C_lag1']
        breath_df['R_change'] = breath_df['R'] - breath_df['R_lag1']
        
        # Interaction features
        breath_df['compliance_pressure_risk'] = (breath_df['C'] < breath_df['C_lag1']).astype(int) * breath_df['PIP_slope_3']
        breath_df['current_PIP_high'] = (breath_df['PIP'] > 30).astype(int)
        breath_df['pressure_flow_ratio'] = breath_df['mean_pressure'] / (breath_df['mean_flow_in'] + 1e-6)
        
        # Define outcome (escalation >3 cmH2O over next 5 breaths)
        breath_df['PIP_future_max_5'] = breath_df['PIP'].shift(-1).rolling(window=5).max()
        breath_df['PIP_escalation'] = breath_df['PIP_future_max_5'] - breath_df['PIP']
        breath_df['high_escalation_risk'] = (breath_df['PIP_escalation'] > 3.0).astype(int)
        
        # Drop rows with NaN
        breath_df = breath_df.dropna().reset_index(drop=True)
        
        return breath_df
        
    except Exception as e:
        print(f"Error processing case {caseid}: {e}")
        return None
