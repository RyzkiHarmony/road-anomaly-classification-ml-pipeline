# Feature Separability Analysis

## Ringkasan Data

class
Non-Event     163
Pothole        45
Speed Bump      1

## Top Features by Separability

         feature    kruskal_p  max_abs_cohen_d  mean_abs_cohen_d
    peak_to_peak 9.266549e-09         1.030512          1.030512
 vertical_energy 8.335747e-08         0.718421          0.718421
           score 9.857916e-08         1.128882          1.128882
gyro_roll_energy 1.086921e-07         0.630573          0.630573
     gyro_energy 3.532150e-07         0.574313          0.574313
      peak_mag_g 4.547093e-07         0.988585          0.988585
        vert_jrk 6.189860e-07         0.876810          0.876810
        max_jerk 6.189860e-07         0.876810          0.876810
   peak_gyro_mag 1.327550e-06         1.089275          1.089275
 gyro_yaw_energy 4.228747e-06         0.616388          0.616388

## Interpretasi Cepat

- Kruskal p kecil berarti distribusi fitur berbeda antar kelas.
- |Cohen's d| besar berarti pemisahan antar kelas lebih kuat.
- Fokuskan model pada fitur dengan p kecil dan effect size besar.
