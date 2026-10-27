# Switch-only experiments: gen_audit rows and CPU throughput (2026-10-28)

`tools/gen_audit.py --exp ID --row` (torch seed 0; 4096 rows per length, 1024 at L 8192; groups from 512 blocks x 8 lanes x
512 points). integer / jumps / var_change / seasonal: % of non-flat series; target_*: % of lanes with that covariate in
their group; nan_*: % of points / of lanes with any NaN. cpu_speed_vs_B0: `tools/gen_throughput.py --device cpu`, points/s
ratio vs B0 measured right after, at L = 1024 / 4096 / 8192 (Windows dev machine; G1.1, G3.1a/b rerun alone because the
first pass ran under load; repeat on GPU). G1.2 and G3.3 change only `train:` (generator = B0).

```
config  share_long_cycle  integer  jumps  var_change  seasonal  flat  nan_points  nan_target_lanes  nan_cov_lanes  target_known  target_past  target_any_cov  in_group       cpu_speed_vs_B0
    B0              9.49     3.20  34.49       19.24     45.74  0.37        0.00              0.00           0.00          7.59         3.37            9.25     43.65             1 / 1 / 1
  G1.1              9.49     3.20  34.49       19.24     45.74  0.37        0.00              0.00           0.00         15.75         6.13           20.51     64.33 0.891 / 0.897 / 0.898
 G3.1a             15.00     3.00  36.20       17.92     47.14  0.28        0.00              0.00           0.00          7.40         3.66            8.94     41.75 1.010 / 1.025 / 1.023
 G3.1b             25.00     2.62  37.54       15.32     47.55  0.29        0.00              0.00           0.00          8.08         3.08            9.30     43.82 1.012 / 1.070 / 1.010
  G4.1              9.49     3.20  36.46       19.93     45.13  0.33        0.00              0.00           0.00          7.54         3.61            9.25     40.84 0.871 / 0.882 / 0.813
 G4.2a              9.49     3.19  35.58       19.54     45.78  0.38        0.00              0.00           0.00          8.03         3.54            9.52     41.55 0.959 / 1.035 / 0.960
 G4.2b              9.49     3.20  37.55       20.37     45.08  0.38        0.00              0.00           0.00          7.62         3.66            9.72     42.94 1.023 / 1.065 / 0.988
  G4.3              9.49     3.20  36.50       20.02     45.56  0.44        0.00              0.00           0.00          6.88         3.15            8.81     43.21 0.953 / 1.019 / 0.936
  G4.4              9.49     3.20  35.96       20.04     45.59  0.34        0.00              0.00           0.00          7.71         3.49            9.55     40.97 0.970 / 1.002 / 0.953
  G5.1              9.49     3.21  35.08       19.53     44.72  0.34        0.00              0.00           0.00          8.33         3.61           10.01     45.43 0.611 / 0.525 / 0.683
  G6.1              9.49     3.20  34.49       19.24     45.74  0.37        1.10             29.57          29.03          7.59         3.37            9.25     43.65 1.015 / 1.030 / 0.988
  G6.2              9.49     3.20  34.49       19.24     45.74  0.37        6.16             30.49          29.77          7.59         3.37            9.25     43.65 1.019 / 0.961 / 0.998
  G6.3              9.49     3.20  34.49       19.24     45.74  0.37        6.50             29.44          27.86          7.59         3.37            9.25     43.65 1.025 / 0.984 / 0.985
  G6.4              9.49     3.20  34.20       19.16     45.86  0.37        0.00              0.00           0.00          7.59         3.37            9.25     43.65 1.044 / 1.039 / 1.005
 G7.4a              9.49    10.49  34.53       20.00     44.11  0.34        0.00              0.00           0.00          7.59         3.37            9.25     43.65 0.942 / 0.861 / 0.898
 G7.4b              9.49    20.17  34.58       20.87     41.93  0.28        0.00              0.00           0.00          7.59         3.37            9.25     43.65 0.859 / 0.885 / 0.866
```

- jumps and var_change move only 1-3 points for G4.1-G4.4: these audit features are dominated by other families and are a
  weak break detector; the switch tests check the break / collapse statistics directly.
- G4.3 simulated crop (ft3_main's own layout sampler, 15,890 contexts): 46.3% of breaks land in the last 10-30% of the real
  context, 29.0% outside it, 24.7% elsewhere inside (v0's uniform placement: 15.9 / 22.1 / 61.9). Below the 50% rule, so
  placement moves to crop time in T3 (owner 2026-10-28).
