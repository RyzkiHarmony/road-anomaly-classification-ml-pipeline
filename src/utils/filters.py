import numpy as np

# 2nd-order Butterworth Bandpass (1Hz - 20Hz at 100Hz) coefficients
B_COEFS = [0.19061660009749753, 0.0, -0.38123320019499507, 0.0, 0.19061660009749753]
A_COEFS = [1.0, -2.3350824020765124, 1.9509646897792021, -0.8192636853124013, 0.2066719851665643]

def butterworth_bandpass_iir(x):
    """
    Apply a 2nd-order Butterworth Bandpass filter (1Hz - 20Hz at 100Hz)
    using a causal IIR difference equation.
    
    Parameters:
        x (np.ndarray): 1D input array.
        
    Returns:
        y (np.ndarray): Filtered 1D array.
    """
    n = len(x)
    y = np.zeros(n, dtype=np.float64)
    
    # Store history of past 4 inputs (x) and 4 outputs (y)
    # [n-1, n-2, n-3, n-4]
    x_hist = [0.0, 0.0, 0.0, 0.0]
    y_hist = [0.0, 0.0, 0.0, 0.0]
    
    b0, b1, b2, b3, b4 = B_COEFS
    a1, a2, a3, a4 = A_COEFS[1:]
    
    for i in range(n):
        val = x[i]
        y_val = (b0 * val + b1 * x_hist[0] + b2 * x_hist[1] + b3 * x_hist[2] + b4 * x_hist[3]
                 - (a1 * y_hist[0] + a2 * y_hist[1] + a3 * y_hist[2] + a4 * y_hist[3]))
        
        y[i] = y_val
        
        # Shift history
        x_hist[3] = x_hist[2]
        x_hist[2] = x_hist[1]
        x_hist[1] = x_hist[0]
        x_hist[0] = val
        
        y_hist[3] = y_hist[2]
        y_hist[2] = y_hist[1]
        y_hist[1] = y_hist[0]
        y_hist[0] = y_val
        
    return y

if __name__ == "__main__":
    # Test parity with scipy lfilter
    from scipy.signal import lfilter
    
    # Generate test signal: sum of 0.2Hz, 5Hz, and 45Hz components
    t = np.linspace(0, 10, 1000)
    sig = np.sin(2 * np.pi * 0.2 * t) + np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 45 * t)
    
    scipy_y = lfilter(B_COEFS, A_COEFS, sig)
    our_y = butterworth_bandpass_iir(sig)
    
    max_diff = np.max(np.abs(scipy_y - our_y))
    print(f"Filter Verification -> Max Absolute Difference: {max_diff:.2e}")
    assert max_diff < 1e-12, "Filter output does not match scipy.signal.lfilter!"
    print("Verification Passed successfully!")
