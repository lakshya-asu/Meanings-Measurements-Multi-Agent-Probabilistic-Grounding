# Init pose review sheet (board item 6)

Provenance: generated at repo commit a3f2ce2b6f7f56d665cb51e6381afe6a5dacac3f by src/scripts/collect_init_poses.py.

Seeded run: seed 20260815, K=5, clearance gate 0.15 m. Re-running with the same seed reproduces these exact candidates.

How to review: open each pair's PNGs (paths are relative to datasets/explore-eqa/), check exactly one ACCEPT box per pair (the RECOMMENDED row is the best composite score, override freely), then run the finalize command at the bottom.

Measured envelope of the existing 49 poses (quality bar the candidates are calibrated against):

- clearance from walls and obstacles (m): min 0.00 / p10 0.00 / median 0.05 / max 0.59
- island radius (m): min 1.34 / p10 5.28 / median 7.14 / max 20.23
- openness, best 90 degree cone mean free range (m): min 0.00 / p10 1.71 / median 2.29 / max 3.72
- distance to nearest object bbox (m): min 0.00 / p10 0.00 / median 0.00 / max 6.37
- poses inside an object bbox: 29

## 00023-zepmXAdrpjR_1

Floor band [-0.272, 0.528] of 1 bands (votes 4), anchors at y = 0.72;0.08;1.02;0.12, island 0 (area 50.7 m2, 0.845 of scene).

| cand | recommended | x | y | z | yaw | clearance_m | island_radius_m | openness_m | bbox_dist_m | score | image | ACCEPT | REJECT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cand_0 | RECOMMENDED | 0.4642 | -0.2725 | 8.3491 | 0.1745 | 0.51 | 8.63 | 2.92 | 0.022 | 0.6672 | pose_candidates/00023-zepmXAdrpjR_1/cand_0.png | [ ] | [ ] |
| cand_1 |  | 8.2693 | 0.1275 | 1.3826 | 1.6581 | 0.23 | 8.63 | 1.66 | 0.107 | 0.4375 | pose_candidates/00023-zepmXAdrpjR_1/cand_1.png | [ ] | [ ] |
| cand_2 |  | 0.0878 | 0.1275 | 0.0413 | -1.9199 | 0.478 | 8.63 | 2.38 | 0.0 | 0.5984 | pose_candidates/00023-zepmXAdrpjR_1/cand_2.png | [ ] | [ ] |
| cand_3 |  | 7.2206 | -0.2725 | 8.8701 | 2.1817 | 0.17 | 8.63 | 2.58 | 4.894 | 0.5949 | pose_candidates/00023-zepmXAdrpjR_1/cand_3.png | [ ] | [ ] |
| cand_4 |  | -4.8207 | -0.1982 | 7.7439 | -2.5307 | 0.255 | 8.63 | 2.75 | 2.266 | 0.646 | pose_candidates/00023-zepmXAdrpjR_1/cand_4.png | [ ] | [ ] |

## 00062-ACZZiU6BXLz_3

Floor band [0.089, 0.489] of 3 bands (votes 4/0/0), anchors at y = 0.94;0.36;1.44;1.43, island 0 (area 15.58 m2, 0.502 of scene).

| cand | recommended | x | y | z | yaw | clearance_m | island_radius_m | openness_m | bbox_dist_m | score | image | ACCEPT | REJECT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cand_0 |  | 8.2264 | 0.0893 | 2.2902 | 0.4363 | 0.283 | 6.32 | 2.14 | 0.225 | 0.4507 | pose_candidates/00062-ACZZiU6BXLz_3/cand_0.png | [ ] | [ ] |
| cand_1 |  | 9.4963 | 0.0893 | -3.265 | 2.1817 | 0.282 | 6.32 | 1.37 | 1.365 | 0.45 | pose_candidates/00062-ACZZiU6BXLz_3/cand_1.png | [ ] | [ ] |
| cand_2 | RECOMMENDED | 11.0106 | 0.0893 | 0.6529 | 0.6981 | 0.633 | 6.32 | 2.59 | 0.108 | 0.6235 | pose_candidates/00062-ACZZiU6BXLz_3/cand_2.png | [ ] | [ ] |
| cand_3 |  | 5.3345 | 0.0893 | 2.8211 | -0.5236 | 0.41 | 6.32 | 1.96 | 0.0 | 0.4605 | pose_candidates/00062-ACZZiU6BXLz_3/cand_3.png | [ ] | [ ] |
| cand_4 |  | 7.2231 | 0.0893 | 2.3388 | -0.9599 | 0.162 | 6.32 | 2.26 | 0.266 | 0.4181 | pose_candidates/00062-ACZZiU6BXLz_3/cand_4.png | [ ] | [ ] |

## 00135-HeSYRw7eMtG_1

Floor band [-0.001, 0.399] of 4 bands (votes 0/4/3/0), anchors at y = 1.99;0.62;1.16;1.45, island 0 (area 25.88 m2, 0.29 of scene).

| cand | recommended | x | y | z | yaw | clearance_m | island_radius_m | openness_m | bbox_dist_m | score | image | ACCEPT | REJECT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cand_0 |  | 3.281 | -0.001 | -4.7336 | 2.1817 | 0.265 | 9.02 | 2.91 | 0.136 | 0.4686 | pose_candidates/00135-HeSYRw7eMtG_1/cand_0.png | [ ] | [ ] |
| cand_1 | RECOMMENDED | -4.0317 | 0.1333 | -5.1446 | -1.8326 | 0.272 | 9.02 | 2.89 | 0.867 | 0.543 | pose_candidates/00135-HeSYRw7eMtG_1/cand_1.png | [ ] | [ ] |
| cand_2 |  | -0.068 | 0.15 | -0.0241 | 0.6981 | 0.177 | 9.02 | 3.29 | 0.0 | 0.4288 | pose_candidates/00135-HeSYRw7eMtG_1/cand_2.png | [ ] | [ ] |
| cand_3 |  | 6.624 | -0.001 | -5.5719 | 0.8727 | 0.159 | 9.02 | 1.61 | 2.54 | 0.3822 | pose_candidates/00135-HeSYRw7eMtG_1/cand_3.png | [ ] | [ ] |
| cand_4 |  | 3.2495 | 0.0903 | -7.1465 | -2.3562 | 0.157 | 9.02 | 1.89 | 0.482 | 0.3586 | pose_candidates/00135-HeSYRw7eMtG_1/cand_4.png | [ ] | [ ] |

## 00149-UuwwmrTsfBN_1

Floor band [0.059, 0.259] of 5 bands (votes 0/4/4/1/0), anchors at y = 0.92;0.59;1.16;0.56, island 0 (area 59.81 m2, 0.349 of scene).

| cand | recommended | x | y | z | yaw | clearance_m | island_radius_m | openness_m | bbox_dist_m | score | image | ACCEPT | REJECT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cand_0 | RECOMMENDED | -0.0837 | 0.0587 | -10.0119 | -0.6981 | 0.537 | 9.56 | 3.34 | 0.0 | 0.5847 | pose_candidates/00149-UuwwmrTsfBN_1/cand_0.png | [ ] | [ ] |
| cand_1 |  | 4.5706 | 0.0587 | -3.3632 | 0.5236 | 0.257 | 9.56 | 2.14 | 0.442 | 0.4314 | pose_candidates/00149-UuwwmrTsfBN_1/cand_1.png | [ ] | [ ] |
| cand_2 |  | -4.0234 | 0.0587 | -3.7214 | 0.6981 | 0.206 | 9.56 | 1.33 | 0.0 | 0.285 | pose_candidates/00149-UuwwmrTsfBN_1/cand_2.png | [ ] | [ ] |
| cand_3 |  | 6.5671 | 0.0587 | -12.8819 | 1.5708 | 0.296 | 9.56 | 1.92 | 5.221 | 0.4802 | pose_candidates/00149-UuwwmrTsfBN_1/cand_3.png | [ ] | [ ] |
| cand_4 |  | -5.1054 | 0.0587 | -8.931 | -1.0472 | 0.168 | 9.56 | 3.01 | 2.883 | 0.537 | pose_candidates/00149-UuwwmrTsfBN_1/cand_4.png | [ ] | [ ] |

## 00207-FRQ75PjD278_0

Floor band [0.088, 1.088] of 1 bands (votes 4), anchors at y = 0.46;0.24;1.51;1.61, island 1 (area 12.45 m2, 0.493 of scene).

| cand | recommended | x | y | z | yaw | clearance_m | island_radius_m | openness_m | bbox_dist_m | score | image | ACCEPT | REJECT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cand_0 |  | 3.2162 | 0.0883 | 5.2486 | -2.2689 | 0.17 | 5.03 | 1.92 | 0.421 | 0.4007 | pose_candidates/00207-FRQ75PjD278_0/cand_0.png | [ ] | [ ] |
| cand_1 |  | -2.9242 | 0.0883 | 6.1138 | 0.0 | 0.151 | 5.03 | 1.47 | 0.278 | 0.3341 | pose_candidates/00207-FRQ75PjD278_0/cand_1.png | [ ] | [ ] |
| cand_2 |  | 4.8085 | 0.0883 | 2.8571 | 1.4835 | 0.161 | 5.03 | 1.99 | 0.006 | 0.3622 | pose_candidates/00207-FRQ75PjD278_0/cand_2.png | [ ] | [ ] |
| cand_3 | RECOMMENDED | 5.589 | 0.0883 | 5.6143 | 1.5708 | 0.228 | 5.03 | 2.01 | 1.619 | 0.491 | pose_candidates/00207-FRQ75PjD278_0/cand_3.png | [ ] | [ ] |
| cand_4 |  | -2.7691 | 0.0883 | 4.8728 | -0.8727 | 0.177 | 5.03 | 1.54 | 0.123 | 0.3356 | pose_candidates/00207-FRQ75PjD278_0/cand_4.png | [ ] | [ ] |

## 00217-qz3829g1Lzf_0

Floor band [0.034, 0.234] of 2 bands (votes 2/0), anchors at y = 0.25;1.38, island 0 (area 20.32 m2, 0.482 of scene).

| cand | recommended | x | y | z | yaw | clearance_m | island_radius_m | openness_m | bbox_dist_m | score | image | ACCEPT | REJECT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cand_0 |  | 3.36 | 0.034 | 4.8572 | 0.5236 | 0.224 | 6.39 | 3.34 | 0.0 | 0.486 | pose_candidates/00217-qz3829g1Lzf_0/cand_0.png | [ ] | [ ] |
| cand_1 |  | -3.8192 | 0.034 | 4.5113 | -0.2618 | 0.28 | 6.39 | 2.04 | 0.182 | 0.4305 | pose_candidates/00217-qz3829g1Lzf_0/cand_1.png | [ ] | [ ] |
| cand_2 |  | 0.066 | 0.034 | -0.0149 | -2.618 | 0.334 | 6.39 | 2.87 | 0.0 | 0.5168 | pose_candidates/00217-qz3829g1Lzf_0/cand_2.png | [ ] | [ ] |
| cand_3 |  | 4.4481 | 0.034 | 3.8999 | 1.6581 | 0.231 | 6.39 | 3.5 | 0.316 | 0.5204 | pose_candidates/00217-qz3829g1Lzf_0/cand_3.png | [ ] | [ ] |
| cand_4 | RECOMMENDED | 4.0707 | 0.034 | 4.481 | 1.3963 | 0.431 | 6.39 | 3.39 | 0.0 | 0.5689 | pose_candidates/00217-qz3829g1Lzf_0/cand_4.png | [ ] | [ ] |

## 00217-qz3829g1Lzf_3

Floor band [0.034, 0.234] of 2 bands (votes 4/0), anchors at y = 1.61;0.86;0.53;0.07, island 0 (area 20.32 m2, 0.482 of scene).

| cand | recommended | x | y | z | yaw | clearance_m | island_radius_m | openness_m | bbox_dist_m | score | image | ACCEPT | REJECT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cand_0 |  | 3.9675 | 0.034 | 4.9596 | 0.7854 | 0.293 | 6.39 | 3.53 | 0.14 | 0.5275 | pose_candidates/00217-qz3829g1Lzf_3/cand_0.png | [ ] | [ ] |
| cand_1 |  | -3.9467 | 0.034 | 4.5993 | -0.2618 | 0.182 | 6.39 | 2.11 | 0.27 | 0.4066 | pose_candidates/00217-qz3829g1Lzf_3/cand_1.png | [ ] | [ ] |
| cand_2 |  | 4.8514 | 0.034 | 4.035 | 1.5708 | 0.155 | 6.39 | 3.53 | 0.533 | 0.5115 | pose_candidates/00217-qz3829g1Lzf_3/cand_2.png | [ ] | [ ] |
| cand_3 |  | -3.1726 | 0.034 | 4.4834 | 0.1745 | 0.229 | 6.39 | 2.03 | 0.154 | 0.4059 | pose_candidates/00217-qz3829g1Lzf_3/cand_3.png | [ ] | [ ] |
| cand_4 | RECOMMENDED | 4.0487 | 0.034 | 4.2945 | 1.4835 | 0.54 | 6.39 | 3.53 | 0.0 | 0.6123 | pose_candidates/00217-qz3829g1Lzf_3/cand_4.png | [ ] | [ ] |

## 00245-741Fdj7NLF9_0

Floor band [-0.066, 0.134] of 1 bands (votes 4), anchors at y = 0.96;0.21;0.46;0.19, island 0 (area 40.2 m2, 1.0 of scene).

| cand | recommended | x | y | z | yaw | clearance_m | island_radius_m | openness_m | bbox_dist_m | score | image | ACCEPT | REJECT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cand_0 |  | 4.3375 | 0.1338 | 15.191 | -2.2689 | 0.358 | 10.31 | 2.24 | 3.615 | 0.6669 | pose_candidates/00245-741Fdj7NLF9_0/cand_0.png | [ ] | [ ] |
| cand_1 |  | -4.8276 | -0.0384 | 11.9646 | -2.0944 | 0.254 | 10.31 | 2.95 | 3.94 | 0.6963 | pose_candidates/00245-741Fdj7NLF9_0/cand_1.png | [ ] | [ ] |
| cand_2 |  | 11.346 | -0.0223 | 12.5682 | 2.5307 | 0.323 | 10.31 | 3.72 | 9.788 | 0.7291 | pose_candidates/00245-741Fdj7NLF9_0/cand_2.png | [ ] | [ ] |
| cand_3 | RECOMMENDED | 11.0628 | 0.0109 | 17.9426 | 0.0873 | 0.519 | 10.31 | 3.04 | 10.529 | 0.8077 | pose_candidates/00245-741Fdj7NLF9_0/cand_3.png | [ ] | [ ] |
| cand_4 |  | -1.0823 | 0.1338 | 14.4309 | 1.6581 | 0.331 | 10.31 | 3.07 | 0.195 | 0.6518 | pose_candidates/00245-741Fdj7NLF9_0/cand_4.png | [ ] | [ ] |

## 00256-92vYG1q49FY_0

Floor band [-2.837, -2.037] of 3 bands (votes 4/2/1), anchors at y = -0.61;-0.40;-1.72;-2.24, island 0 (area 38.79 m2, 0.548 of scene).

| cand | recommended | x | y | z | yaw | clearance_m | island_radius_m | openness_m | bbox_dist_m | score | image | ACCEPT | REJECT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cand_0 |  | -0.6722 | -2.6367 | -0.0638 | -0.2618 | 0.234 | 8.95 | 2.79 | 0.0 | 0.482 | pose_candidates/00256-92vYG1q49FY_0/cand_0.png | [ ] | [ ] |
| cand_1 |  | 3.6639 | -2.6367 | 6.7782 | 1.309 | 0.234 | 8.95 | 2.59 | 1.348 | 0.5624 | pose_candidates/00256-92vYG1q49FY_0/cand_1.png | [ ] | [ ] |
| cand_2 |  | -5.5042 | -2.6367 | 6.3493 | -0.6981 | 0.195 | 8.95 | 1.18 | 2.438 | 0.4061 | pose_candidates/00256-92vYG1q49FY_0/cand_2.png | [ ] | [ ] |
| cand_3 | RECOMMENDED | -6.7845 | -2.8367 | -4.9131 | -2.9671 | 0.243 | 8.95 | 2.72 | 2.517 | 0.579 | pose_candidates/00256-92vYG1q49FY_0/cand_3.png | [ ] | [ ] |
| cand_4 |  | 0.088 | -2.8367 | -5.2666 | 2.3562 | 0.401 | 8.95 | 3.89 | 0.0 | 0.5701 | pose_candidates/00256-92vYG1q49FY_0/cand_4.png | [ ] | [ ] |

## 00262-1xGrZPxG1Hz_0

Floor band [0.111, 0.711] of 3 bands (votes 4/2/0), anchors at y = 0.51;1.57;1.49;0.60, island 1 (area 23.75 m2, 0.548 of scene).

| cand | recommended | x | y | z | yaw | clearance_m | island_radius_m | openness_m | bbox_dist_m | score | image | ACCEPT | REJECT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cand_0 |  | -6.9448 | 0.1108 | 2.8765 | 1.9199 | 0.249 | 6.09 | 2.05 | 0.374 | 0.4518 | pose_candidates/00262-1xGrZPxG1Hz_0/cand_0.png | [ ] | [ ] |
| cand_1 |  | -8.8153 | 0.1108 | 0.0017 | -2.5307 | 0.308 | 6.09 | 2.24 | 0.0 | 0.4567 | pose_candidates/00262-1xGrZPxG1Hz_0/cand_1.png | [ ] | [ ] |
| cand_2 | RECOMMENDED | -3.7927 | 0.1108 | 2.9799 | -0.6109 | 0.506 | 6.09 | 3.25 | 0.075 | 0.6195 | pose_candidates/00262-1xGrZPxG1Hz_0/cand_2.png | [ ] | [ ] |
| cand_3 |  | -8.923 | 0.1108 | 2.4806 | -2.3562 | 0.224 | 6.09 | 2.46 | 1.736 | 0.545 | pose_candidates/00262-1xGrZPxG1Hz_0/cand_3.png | [ ] | [ ] |
| cand_4 |  | -6.9978 | 0.1108 | -0.0241 | 1.7453 | 0.169 | 6.09 | 1.76 | 0.0 | 0.3534 | pose_candidates/00262-1xGrZPxG1Hz_0/cand_4.png | [ ] | [ ] |

## 00262-1xGrZPxG1Hz_5

Floor band [0.111, 0.711] of 3 bands (votes 4/3/0), anchors at y = 0.47;1.34;1.68;1.54, island 1 (area 23.75 m2, 0.548 of scene).

| cand | recommended | x | y | z | yaw | clearance_m | island_radius_m | openness_m | bbox_dist_m | score | image | ACCEPT | REJECT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cand_0 |  | -6.6924 | 0.1108 | 3.0028 | 1.9199 | 0.375 | 6.09 | 2.18 | 0.156 | 0.4937 | pose_candidates/00262-1xGrZPxG1Hz_5/cand_0.png | [ ] | [ ] |
| cand_1 |  | -8.8838 | 0.1108 | -0.0991 | -2.5307 | 0.237 | 6.09 | 2.29 | 0.0 | 0.4332 | pose_candidates/00262-1xGrZPxG1Hz_5/cand_1.png | [ ] | [ ] |
| cand_2 | RECOMMENDED | -3.6451 | 0.1108 | 3.418 | -0.4363 | 0.653 | 6.09 | 3.41 | 0.0 | 0.6706 | pose_candidates/00262-1xGrZPxG1Hz_5/cand_2.png | [ ] | [ ] |
| cand_3 |  | -8.9956 | 0.1108 | 2.2876 | -2.4435 | 0.151 | 6.09 | 2.62 | 1.543 | 0.5318 | pose_candidates/00262-1xGrZPxG1Hz_5/cand_3.png | [ ] | [ ] |
| cand_4 |  | -7.0827 | 0.1108 | -0.0334 | 1.7453 | 0.162 | 6.09 | 1.82 | 0.0 | 0.3558 | pose_candidates/00262-1xGrZPxG1Hz_5/cand_4.png | [ ] | [ ] |

## 00313-PE6kVEtrxtj_0

Floor band [0.099, 0.299] of 2 bands (votes 4/0), anchors at y = 0.70;0.76;0.63;0.86, island 0 (area 23.8 m2, 0.565 of scene).

| cand | recommended | x | y | z | yaw | clearance_m | island_radius_m | openness_m | bbox_dist_m | score | image | ACCEPT | REJECT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cand_0 | RECOMMENDED | 0.087 | 0.0985 | 6.0882 | 1.1345 | 0.635 | 5.88 | 3.05 | 0.0 | 0.667 | pose_candidates/00313-PE6kVEtrxtj_0/cand_0.png | [ ] | [ ] |
| cand_1 |  | 4.307 | 0.0985 | 5.3962 | 1.1345 | 0.276 | 5.88 | 1.62 | 0.509 | 0.4361 | pose_candidates/00313-PE6kVEtrxtj_0/cand_1.png | [ ] | [ ] |
| cand_2 |  | -3.6753 | 0.1302 | 5.7962 | -2.0071 | 0.158 | 5.88 | 3.34 | 0.544 | 0.5306 | pose_candidates/00313-PE6kVEtrxtj_0/cand_2.png | [ ] | [ ] |
| cand_3 |  | -2.4353 | 0.1995 | 7.3255 | -0.0873 | 0.262 | 5.88 | 3.08 | 0.279 | 0.5456 | pose_candidates/00313-PE6kVEtrxtj_0/cand_3.png | [ ] | [ ] |
| cand_4 |  | -2.1771 | 0.0985 | 5.752 | -1.309 | 0.308 | 5.88 | 3.63 | 0.021 | 0.5382 | pose_candidates/00313-PE6kVEtrxtj_0/cand_4.png | [ ] | [ ] |

## 00366-fxbzYAGkrtm_1

Floor band [-0.783, 0.817] of 2 bands (votes 0/4), anchors at y = 0.31;0.20;0.95;0.38, island 0 (area 58.58 m2, 0.55 of scene).

| cand | recommended | x | y | z | yaw | clearance_m | island_radius_m | openness_m | bbox_dist_m | score | image | ACCEPT | REJECT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cand_0 |  | -1.816 | 0.0174 | 6.0886 | 2.7053 | 0.3 | 10.25 | 1.91 | 0.124 | 0.433 | pose_candidates/00366-fxbzYAGkrtm_1/cand_0.png | [ ] | [ ] |
| cand_1 |  | 3.9967 | 0.0174 | -1.4787 | 1.4835 | 0.351 | 10.25 | 2.04 | 0.415 | 0.4958 | pose_candidates/00366-fxbzYAGkrtm_1/cand_1.png | [ ] | [ ] |
| cand_2 |  | -10.8034 | 0.0174 | 5.6624 | -1.5708 | 0.234 | 10.25 | 1.51 | 5.447 | 0.455 | pose_candidates/00366-fxbzYAGkrtm_1/cand_2.png | [ ] | [ ] |
| cand_3 |  | 3.3863 | 0.0174 | 11.6944 | 0.8727 | 0.214 | 10.25 | 2.13 | 2.106 | 0.5085 | pose_candidates/00366-fxbzYAGkrtm_1/cand_3.png | [ ] | [ ] |
| cand_4 | RECOMMENDED | -2.702 | 0.107 | 12.1909 | -0.8727 | 0.494 | 10.25 | 2.89 | 2.406 | 0.697 | pose_candidates/00366-fxbzYAGkrtm_1/cand_4.png | [ ] | [ ] |

## 00388-pcpn6mFqFCg_0

Floor band [0.154, 0.754] of 2 bands (votes 2/4), anchors at y = 0.50;0.26;0.75;0.31, island 0 (area 5.32 m2, 0.167 of scene).

| cand | recommended | x | y | z | yaw | clearance_m | island_radius_m | openness_m | bbox_dist_m | score | image | ACCEPT | REJECT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cand_0 |  | 10.914 | 0.1543 | 1.1564 | 3.1416 | 0.345 | 3.78 | 1.58 | 0.0 | 0.3294 | pose_candidates/00388-pcpn6mFqFCg_0/cand_0.png | [ ] | [ ] |
| cand_1 |  | 12.4879 | 0.1543 | -1.3717 | 1.8326 | 0.157 | 3.78 | 1.53 | 0.0 | 0.2487 | pose_candidates/00388-pcpn6mFqFCg_0/cand_1.png | [ ] | [ ] |
| cand_2 |  | 10.4652 | 0.1543 | 3.2422 | -0.8727 | 0.185 | 3.78 | 1.71 | 1.561 | 0.3787 | pose_candidates/00388-pcpn6mFqFCg_0/cand_2.png | [ ] | [ ] |
| cand_3 |  | 10.7129 | 0.1543 | -1.5271 | -1.9199 | 0.201 | 3.78 | 1.42 | 0.0 | 0.256 | pose_candidates/00388-pcpn6mFqFCg_0/cand_3.png | [ ] | [ ] |
| cand_4 | RECOMMENDED | 11.0026 | 0.1543 | 2.2276 | 0.0 | 0.236 | 3.78 | 1.8 | 1.006 | 0.4081 | pose_candidates/00388-pcpn6mFqFCg_0/cand_4.png | [ ] | [ ] |

## 00720-8B43pG641ff_0

Floor band [0.053, 1.053] of 5 bands (votes 0/1/4/0/0), anchors at y = 1.93;1.99;1.16;2.04, island 0 (area 57.33 m2, 0.565 of scene).

| cand | recommended | x | y | z | yaw | clearance_m | island_radius_m | openness_m | bbox_dist_m | score | image | ACCEPT | REJECT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cand_0 |  | -3.1477 | 0.053 | 6.079 | 2.0071 | 0.257 | 10.08 | 2.34 | 0.506 | 0.5007 | pose_candidates/00720-8B43pG641ff_0/cand_0.png | [ ] | [ ] |
| cand_1 |  | -8.4689 | 0.053 | -1.4955 | -2.9671 | 0.285 | 10.08 | 2.14 | 0.001 | 0.4414 | pose_candidates/00720-8B43pG641ff_0/cand_1.png | [ ] | [ ] |
| cand_2 |  | 4.8887 | 0.053 | 9.1556 | 0.6109 | 0.161 | 10.08 | 1.87 | 3.968 | 0.4643 | pose_candidates/00720-8B43pG641ff_0/cand_2.png | [ ] | [ ] |
| cand_3 |  | -0.0865 | 0.053 | -0.4005 | 2.7053 | 0.21 | 10.08 | 3.08 | 0.0 | 0.497 | pose_candidates/00720-8B43pG641ff_0/cand_3.png | [ ] | [ ] |
| cand_4 | RECOMMENDED | -8.7318 | 0.053 | 10.488 | -1.0472 | 0.158 | 10.08 | 2.28 | 6.868 | 0.5039 | pose_candidates/00720-8B43pG641ff_0/cand_4.png | [ ] | [ ] |

## Finalize

After checking one ACCEPT box per pair, run inside the container:

    python3 -m src.scripts.collect_init_poses finalize --from-review

This writes datasets/explore-eqa/scene_init_poses_semantic_only_v2.csv (the original 49 rows verbatim plus the approved 15) and prints its sha256 for splits/MANIFEST.json. The original CSV is never edited.

Approved by: ____________  Date: ____________
