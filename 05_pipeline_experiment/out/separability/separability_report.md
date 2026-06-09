# Feature Separability Analysis

## Ringkasan Data

class
Non-Event     3331
Pothole        264
Speed Bump      96

## Top Features by Separability

             feature     kruskal_p  max_abs_cohen_d  mean_abs_cohen_d
            max_jerk 1.469090e-152         2.475628          1.608995
        peak_to_peak 6.533404e-151         2.404824          1.583593
     hjorth_activity 4.298015e-134         2.071857          1.377685
     vertical_energy 2.273653e-116         1.870835          1.245236
speed_normalized_p2p 2.250463e-115         1.953291          1.192823
  linear_jerk_3d_max  3.726224e-93         1.422281          0.959472
    gyro_roll_energy  1.095856e-83         1.123215          0.723405
        snr_vertical  5.485262e-83         2.388970          1.392550
        crest_factor  8.073220e-71         1.243849          0.828989
   gyro_pitch_energy  1.221545e-68         0.853963          0.552895

## Interpretasi Cepat

- Kruskal p kecil berarti distribusi fitur berbeda antar kelas.
- |Cohen's d| besar berarti pemisahan antar kelas lebih kuat.
- Fokuskan model pada fitur dengan p kecil dan effect size besar.
