# Feature Separability Analysis

## Ringkasan Data

class
Non-Event      28
Pothole         2
Speed Bump    100

## Top Features by Separability

                     feature  kruskal_p  max_abs_cohen_d  mean_abs_cohen_d
horizontal_to_vertical_ratio   0.001402         1.070035          0.725614
                    kurtosis   0.064264         1.478716          0.887493
                  grav_z_std   0.073988         0.704616          0.413650
             num_peaks_accel   0.094801         1.489296          0.872838
                    vert_jrk   0.114801         3.755770          1.738068
          peak_interval_mean   0.138316         0.941120          0.656953
            gyro_roll_energy   0.138784         1.498905          0.887155
          linear_jerk_3d_max   0.402089         0.669730          0.440786
                  grav_y_std   0.402558         0.452156          0.321162
                    skewness   0.522792         1.101425          0.733293

## Interpretasi Cepat

- Kruskal p kecil berarti distribusi fitur berbeda antar kelas.
- |Cohen's d| besar berarti pemisahan antar kelas lebih kuat.
- Fokuskan model pada fitur dengan p kecil dan effect size besar.
