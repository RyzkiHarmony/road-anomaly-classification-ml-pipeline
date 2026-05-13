# Feature Separability Analysis

## Ringkasan Data

class
Non-Event     494
Pothole       134
Speed Bump      5

## Top Features by Separability

         feature    kruskal_p  max_abs_cohen_d  mean_abs_cohen_d
    peak_to_peak 1.923299e-25         1.382296          0.932524
           score 1.652626e-24         1.665957          1.096869
      peak_mag_g 2.043523e-22         1.137059          0.762688
        vert_jrk 7.914774e-22         1.743053          1.199330
        max_jerk 7.914774e-22         1.743053          1.199330
 vertical_energy 1.220700e-20         1.156091          0.839647
 gyro_yaw_energy 7.876391e-19         0.656868          0.441344
     gyro_energy 2.353405e-17         0.633590          0.461136
        kurtosis 6.665747e-17         0.924242          0.590951
gyro_roll_energy 8.013310e-16         0.745405          0.549979

## Interpretasi Cepat

- Kruskal p kecil berarti distribusi fitur berbeda antar kelas.
- |Cohen's d| besar berarti pemisahan antar kelas lebih kuat.
- Fokuskan model pada fitur dengan p kecil dan effect size besar.
