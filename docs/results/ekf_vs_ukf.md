| condition | scenario | estimator | n | success_rate | dock_pos_err | dock_yaw_err_deg | est_rmse_xy | nees_mean | track_e2_rmse | cross_track_rmse | effort | duration |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| default | nominal | ekf | 10 | 100 % | 2.04 ± 1.01 cm | 2.16 ± 1.31 | 8.71 ± 4.04 cm | 1.93 ± 1.04 | 2.73 ± 1.82 cm | 7.89 ± 5.34 cm | 0.38 ± 0.02 | 40.72 ± 3.53 |
| default | nominal | ukf | 10 | 100 % | 1.97 ± 0.94 cm | 2.07 ± 1.20 | 8.64 ± 4.19 cm | 1.80 ± 0.80 | 2.70 ± 1.85 cm | 7.88 ± 5.40 cm | 0.38 ± 0.02 | 41.02 ± 3.54 |
| default | stress | ekf | 10 | 100 % | 2.63 ± 1.25 cm | 2.69 ± 1.50 | 9.85 ± 3.40 cm | 12.54 ± 7.40 | 4.13 ± 1.58 cm | 8.91 ± 4.37 cm | 0.41 ± 0.03 | 40.20 ± 4.52 |
| default | stress | ukf | 10 | 90 % | 11.13 ± 26.06 cm | 4.61 ± 6.23 | 13.83 ± 12.72 cm | 17.00 ± 10.20 | 4.30 ± 1.65 cm | 10.77 ± 7.24 cm | 0.40 ± 0.03 | 42.02 ± 5.62 |
