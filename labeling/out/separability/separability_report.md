# Feature Separability Analysis

## Ringkasan Data

class
Non-Event     48
Pothole       14
Speed Bump     3

## Top Features by Separability

            feature    kruskal_p  max_abs_cohen_d  mean_abs_cohen_d
              score 1.070633e-07         4.812273          2.959073
      peak_gyro_mag 1.100141e-04         1.508961          1.076991
         peak_mag_g 9.932965e-04         1.332654          0.909659
        gyro_energy 8.111968e-03         1.352845          0.964254
       accel_energy 9.691630e-03         1.632697          1.103263
         speed_mean 1.870668e-02         0.810660          0.574227
accel_to_gyro_ratio 1.052169e-01         0.918928          0.656563
 peak_interval_mean 3.028118e-01         0.455260          0.316761
            mag_jrk 3.049434e-01         1.257897          0.727391
    asymmetry_score 4.623386e-01         0.555194          0.385744

## Interpretasi Cepat

- Kruskal p kecil berarti distribusi fitur berbeda antar kelas.
- |Cohen's d| besar berarti pemisahan antar kelas lebih kuat.
- Fokuskan model pada fitur dengan p kecil dan effect size besar.
