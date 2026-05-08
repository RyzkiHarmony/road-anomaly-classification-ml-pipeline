# Feature Separability Analysis

## Ringkasan Data

class
Non-Event     29
Pothole        9
Speed Bump     4

## Top Features by Separability

            feature  kruskal_p  max_abs_cohen_d  mean_abs_cohen_d
              score   0.003333         1.665739          1.150831
           max_jerk   0.005334         1.290541          0.850128
           kurtosis   0.005449         1.688651          1.035414
       peak_to_peak   0.016127         1.529511          0.974187
 fft_high_low_ratio   0.017517         0.858700          0.610047
      peak_gyro_mag   0.025159         1.288198          0.803219
         peak_mag_g   0.026294         1.130430          0.789186
accel_to_gyro_ratio   0.032345         0.934253          0.637706
                zcr   0.039865         0.893538          0.582028
   gyro_roll_energy   0.124892         1.015661          0.663422

## Interpretasi Cepat

- Kruskal p kecil berarti distribusi fitur berbeda antar kelas.
- |Cohen's d| besar berarti pemisahan antar kelas lebih kuat.
- Fokuskan model pada fitur dengan p kecil dan effect size besar.
