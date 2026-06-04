# Feature Separability Analysis

## Ringkasan Data

class
Non-Event     649
Pothole        53
Speed Bump     93

## Top Features by Separability

                     feature    kruskal_p  max_abs_cohen_d  mean_abs_cohen_d
                    vert_jrk 3.203907e-46         2.483893          1.532918
        speed_normalized_p2p 3.230814e-38         1.550610          0.964084
                    kurtosis 1.510127e-31         1.232691          0.813720
horizontal_to_vertical_ratio 2.401246e-27         1.061394          0.733116
          linear_jerk_3d_max 2.325116e-25         1.417852          0.945109
            gyro_roll_energy 6.510961e-24         1.377291          0.947186
                    peak_mag 2.165768e-18         1.277793          0.834852
             num_peaks_accel 1.532475e-03         0.336420          0.237850
              event_duration 1.638440e-03         0.326951          0.198469
                  grav_z_std 3.230024e-03         0.180934          0.129687

## Interpretasi Cepat

- Kruskal p kecil berarti distribusi fitur berbeda antar kelas.
- |Cohen's d| besar berarti pemisahan antar kelas lebih kuat.
- Fokuskan model pada fitur dengan p kecil dan effect size besar.
