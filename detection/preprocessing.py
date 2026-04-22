import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

class Preprocessor:
    def __init__(self, sampling_rate=50):
        """
        Initialize the Preprocessor.
        :param sampling_rate: Sampling rate of the sensor data in Hz.
        """
        self.sampling_rate = sampling_rate

    def load_data(self, file_path):
        """
        Load data from a CSV file.
        :param file_path: Path to the CSV file.
        :return: DataFrame containing the sensor data.
        """
        try:
            df = pd.read_csv(file_path)
            # Ensure required columns exist
            required_cols = ['timestamp', 'accelX', 'accelY', 'accelZ', 'latitude', 'longitude']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Missing required columns. Expected: {required_cols}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def apply_low_pass_filter(self, data, cutoff=5.0, order=4):
        """
        Apply a Low-Pass Butterworth filter to remove high-frequency noise.
        :param data: Input signal (array-like).
        :param cutoff: Cutoff frequency in Hz.
        :param order: Order of the filter.
        :return: Filtered signal.
        """
        nyquist = 0.5 * self.sampling_rate
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    def reorient_acceleration(self, df):
        """
        Align accelerometer data to a global coordinate system.
        Since we only have accelerometer data, we estimate gravity from the low-frequency component.
        Vertical acceleration is the projection of total acceleration onto the gravity vector.
        
        :param df: DataFrame with accelX, accelY, accelZ.
        :return: DataFrame with added 'accel_vertical' column.
        """
        # 1. Estimate gravity vector using a low-pass filter (very low cutoff)
        # Gravity is constant/slow-changing compared to vibrations
        gravity_x = self.apply_low_pass_filter(df['accelX'], cutoff=1.0)
        gravity_y = self.apply_low_pass_filter(df['accelY'], cutoff=1.0)
        gravity_z = self.apply_low_pass_filter(df['accelZ'], cutoff=1.0)

        # 2. Calculate vertical acceleration (projection onto gravity vector)
        # a_v = (a . g) / |g|
        # Note: We subtract gravity magnitude (approx 9.8 or |g|) to get dynamic vertical acceleration
        
        vertical_acc = []
        for i in range(len(df)):
            g_vec = np.array([gravity_x[i], gravity_y[i], gravity_z[i]])
            a_vec = np.array([df['accelX'].iloc[i], df['accelY'].iloc[i], df['accelZ'].iloc[i]])
            
            g_norm = np.linalg.norm(g_vec)
            if g_norm == 0:
                vertical_acc.append(0.0)
            else:
                # Project acceleration onto gravity direction
                proj = np.dot(a_vec, g_vec) / g_norm
                # Remove static gravity component
                vertical_acc.append(proj - g_norm)

        df['accel_vertical'] = vertical_acc
        
        # Apply filter to vertical acceleration as well to remove high freq noise
        df['accel_vertical_filtered'] = self.apply_low_pass_filter(df['accel_vertical'], cutoff=10.0)
        
        return df

    def create_windows(self, df, window_size_sec=2.0, overlap_percent=0.5):
        """
        Segment the continuous stream into sliding windows.
        :param df: DataFrame containing the data.
        :param window_size_sec: Size of the window in seconds.
        :param overlap_percent: Overlap percentage (0.0 to 1.0).
        :return: List of DataFrame windows.
        """
        window_size_samples = int(window_size_sec * self.sampling_rate)
        step_size = int(window_size_samples * (1 - overlap_percent))
        
        windows = []
        num_samples = len(df)
        
        for i in range(0, num_samples - window_size_samples + 1, step_size):
            window = df.iloc[i : i + window_size_samples].copy()
            # Check if window time consistency is valid (optional, but good for real data)
            # Assuming data is continuous for now
            windows.append(window)
            
        return windows

if __name__ == "__main__":
    # Example usage
    # Create dummy data for testing
    import numpy as np
    
    dates = pd.date_range(start='2023-01-01', periods=200, freq='20ms') # 50Hz
    data = {
        'timestamp': dates.astype(np.int64) // 10**6,
        'accelX': np.random.normal(0, 0.5, 200),
        'accelY': np.random.normal(0, 0.5, 200),
        'accelZ': np.random.normal(9.8, 0.5, 200), # Gravity on Z
        'latitude': np.linspace(-6.2, -6.21, 200),
        'longitude': np.linspace(106.8, 106.81, 200)
    }
    df = pd.DataFrame(data)
    
    preprocessor = Preprocessor(sampling_rate=50)
    df_reoriented = preprocessor.reorient_acceleration(df)
    windows = preprocessor.create_windows(df_reoriented)
    
    print(f"Data shape: {df.shape}")
    print(f"Number of windows: {len(windows)}")
    if len(windows) > 0:
        print(f"First window shape: {windows[0].shape}")
        print(windows[0][['accel_vertical', 'accel_vertical_filtered']].head())
