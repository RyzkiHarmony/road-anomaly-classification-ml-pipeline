# Feature Separability Analysis

## Ringkasan Data

class
Non-Event     4292
Pothole        285
Speed Bump     249

## Top Features by Separability

                     feature     kruskal_p  max_abs_cohen_d  mean_abs_cohen_d
             hjorth_activity 3.750241e-200         2.236675          1.459400
          linear_jerk_3d_max 3.484161e-129         1.504145          0.997248
                crest_factor 9.436803e-112         1.288463          0.850545
horizontal_to_vertical_ratio 2.589279e-108         1.007834          0.721198
         waveform_complexity  2.096093e-16         0.049739          0.022483
                  grav_y_std  6.324190e-14         0.357362          0.235603
             rise_time_ratio  1.737640e-10         0.129892          0.120109
         first_peak_polarity  7.738244e-02         0.137863          0.092036

## Interpretasi Cepat

- Kruskal p kecil berarti distribusi fitur berbeda antar kelas.
- |Cohen's d| besar berarti pemisahan antar kelas lebih kuat.
- Fokuskan model pada fitur dengan p kecil dan effect size besar.
