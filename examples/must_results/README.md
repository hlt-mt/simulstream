# Baseline Results for Each Language Pair of MuST-C

Below we report the scores for each language pair for the systems reported [in the paper](https://arxiv.org/abs/2512.17648).
The reported scores represent: Quality (COMET, BLUE), latency (StreamLAAL, StreamLAAL\_CA), flickering (NE), and computational cost (RTF) of retranslation and incremental speech processors (Speech P.) with Canary and SeamlessM4T v1 medium. The column w/f/t refers to varying the window length (w) for sliding window, the number of frames (f) for StreamAtt, and the VAD probability threshold (t) for VAD-based sliding window.

## en-de

| Speech P.     | Method               | Model       | w/f/t | COMET ↑ | BLEU ↑ | StreamLAAL ↓ | StreamLAAL<sub>CA</sub> ↓ |   NE ↓ |  RTF ↓ |
| ------------- | -------------------- | ----------- | ----: | ------: | -----: | -----------: | ------------------------: | -----: | -----: |
| retranslation | Sliding Window       | Canary      |     8 |  0.7591 |  28.12 |         2.56 |                      2.89 | 0.6416 | 0.1444 |
| retranslation | Sliding Window       | Canary      |    10 |  0.7656 |  27.79 |         3.09 |                      3.48 | 0.8750 | 0.1743 |
| retranslation | Sliding Window       | Canary      |    12 |  0.7768 |  28.41 |         3.48 |                      4.14 | 1.0082 | 0.2961 |
| retranslation | Sliding Window       | Canary      |    14 |  0.7814 |  29.01 |         4.07 |                      4.60 | 1.1995 | 0.2361 |
| retranslation | Sliding Window       | SeamlessM4T |     8 |  0.6793 |  22.34 |         2.39 |                      2.77 | 0.8799 | 0.1753 |
| retranslation | Sliding Window       | SeamlessM4T |    10 |  0.6982 |  23.07 |         3.09 |                      3.56 | 1.0688 | 0.2115 |
| retranslation | Sliding Window       | SeamlessM4T |    12 |  0.7104 |  23.63 |         3.69 |                      4.21 | 1.2380 | 0.2405 |
| retranslation | Sliding Window       | SeamlessM4T |    14 |  0.7108 |  23.26 |         4.38 |                      5.13 | 1.3938 | 0.3357 |
| retranslation | VAD + Sliding Window | Canary      |   0.6 |  0.7147 |  24.02 |         2.58 |                      2.82 | 0.1387 | 0.0816 |
| retranslation | VAD + Sliding Window | Canary      |   0.5 |  0.7211 |  24.45 |         2.60 |                      2.85 | 0.1485 | 0.0831 |
| retranslation | VAD + Sliding Window | Canary      |   0.4 |  0.7259 |  24.79 |         2.61 |                      2.87 | 0.1567 | 0.0861 |
| retranslation | VAD + Sliding Window | Canary      |   0.3 |  0.7325 |  25.11 |         2.66 |                      2.94 | 0.1726 | 0.0881 |
| retranslation | VAD + Sliding Window | SeamlessM4T |   0.6 |  0.6700 |  20.45 |         2.55 |                      2.83 | 0.1797 | 0.0949 |
| retranslation | VAD + Sliding Window | SeamlessM4T |   0.5 |  0.6730 |  20.56 |         2.59 |                      2.87 | 0.1857 | 0.0965 |
| retranslation | VAD + Sliding Window | SeamlessM4T |   0.4 |  0.6772 |  20.86 |         2.72 |                      3.01 | 0.2028 | 0.0977 |
| retranslation | VAD + Sliding Window | SeamlessM4T |   0.3 |  0.6792 |  21.27 |         2.76 |                      3.07 | 0.2226 | 0.1033 |
| incremental   | StreamAtt            | SeamlessM4T |     2 |  0.7305 |  25.15 |         1.95 |                      2.23 | 0.0000 | 0.2226 |
| incremental   | StreamAtt            | SeamlessM4T |     4 |  0.7382 |  26.12 |         2.23 |                      2.51 | 0.0000 | 0.2401 |
| incremental   | StreamAtt            | SeamlessM4T |     6 |  0.7420 |  26.55 |         2.53 |                      2.83 | 0.0000 | 0.2562 |
| incremental   | StreamAtt            | SeamlessM4T |     8 |  0.7416 |  26.62 |         2.80 |                      3.09 | 0.0000 | 0.2649 |

## en-es

| Speech P.     | Method               | Model       | w/f/t | COMET ↑ | BLEU ↑ | StreamLAAL ↓ | StreamLAAL<sub>CA</sub> ↓ |   NE ↓ |  RTF ↓ |
| ------------- | -------------------- | ----------- | ----: | ------: | -----: | -----------: | ------------------------: | -----: | -----: |
| retranslation | Sliding Window       | Canary      |     8 |  0.7925 |  32.39 |         2.35 |                      2.67 | 0.5375 | 0.1373 |
| retranslation | Sliding Window       | Canary      |    10 |  0.7938 |  31.99 |         2.96 |                      3.52 | 0.7666 | 0.2466 |
| retranslation | Sliding Window       | Canary      |    12 |  0.7985 |  32.16 |         3.35 |                      3.99 | 0.9083 | 0.2862 |
| retranslation | Sliding Window       | Canary      |    14 |  0.8016 |  32.61 |         3.92 |                      4.40 | 1.1009 | 0.2199 |
| retranslation | Sliding Window       | SeamlessM4T |     8 |  0.7304 |  26.97 |         2.29 |                      2.72 | 0.8658 | 0.2007 |
| retranslation | Sliding Window       | SeamlessM4T |    10 |  0.7495 |  27.92 |         2.69 |                      3.19 | 0.9758 | 0.2301 |
| retranslation | Sliding Window       | SeamlessM4T |    12 |  0.7604 |  28.53 |         3.29 |                      3.87 | 1.1476 | 0.2674 |
| retranslation | Sliding Window       | SeamlessM4T |    14 |  0.7652 |  28.51 |         3.98 |                      4.67 | 1.3073 | 0.3144 |
| retranslation | VAD + Sliding Window | Canary      |   0.6 |  0.7437 |  27.94 |         2.51 |                      2.75 | 0.1169 | 0.0794 |
| retranslation | VAD + Sliding Window | Canary      |   0.5 |  0.7489 |  28.34 |         2.55 |                      2.79 | 0.1240 | 0.0807 |
| retranslation | VAD + Sliding Window | Canary      |   0.4 |  0.7529 |  28.72 |         2.60 |                      2.85 | 0.1408 | 0.0803 |
| retranslation | VAD + Sliding Window | Canary      |   0.3 |  0.7580 |  29.12 |         2.58 |                      2.84 | 0.1550 | 0.0845 |
| retranslation | VAD + Sliding Window | SeamlessM4T |   0.6 |  0.7069 |  24.40 |         2.45 |                      2.72 | 0.1586 | 0.0931 |
| retranslation | VAD + Sliding Window | SeamlessM4T |   0.5 |  0.7099 |  24.55 |         2.51 |                      2.78 | 0.1664 | 0.0937 |
| retranslation | VAD + Sliding Window | SeamlessM4T |   0.4 |  0.7127 |  24.97 |         2.54 |                      2.82 | 0.1745 | 0.0947 |
| retranslation | VAD + Sliding Window | SeamlessM4T |   0.3 |  0.7171 |  25.72 |         2.61 |                      2.91 | 0.1990 | 0.0990 |
| incremental   | StreamAtt            | SeamlessM4T |     2 |  0.7655 |  28.63 |         1.74 |                      1.99 | 0.0000 | 0.2068 |
| incremental   | StreamAtt            | SeamlessM4T |     4 |  0.7741 |  29.70 |         2.10 |                      2.36 | 0.0000 | 0.2211 |
| incremental   | StreamAtt            | SeamlessM4T |     6 |  0.7793 |  30.40 |         2.43 |                      2.69 | 0.0000 | 0.2314 |
| incremental   | StreamAtt            | SeamlessM4T |     8 |  0.7787 |  30.87 |         2.75 |                      3.01 | 0.0000 | 0.2443 |

## en-fr

| Speech P.     | Method               | Model       | w/f/t | COMET ↑ | BLEU ↑ | StreamLAAL ↓ | StreamLAAL<sub>CA</sub> ↓ |   NE ↓ |  RTF ↓ |
| ------------- | -------------------- | ----------- | ----: | ------: | -----: | -----------: | ------------------------: | -----: | -----: |
| retranslation | Sliding Window       | Canary      |     8 |  0.7920 |  38.19 |         2.42 |                      2.78 | 0.5547 | 0.1531 |
| retranslation | Sliding Window       | Canary      |    10 |  0.7929 |  38.46 |         2.92 |                      3.34 | 0.7711 | 0.1846 |
| retranslation | Sliding Window       | Canary      |    12 |  0.7986 |  39.13 |         3.30 |                      3.81 | 0.9097 | 0.2249 |
| retranslation | Sliding Window       | Canary      |    14 |  0.8025 |  39.34 |         3.83 |                      4.56 | 1.0840 | 0.3314 |
| retranslation | Sliding Window       | SeamlessM4T |     8 |  0.7226 |  32.98 |         2.43 |                      2.81 | 0.8330 | 0.1776 |
| retranslation | Sliding Window       | SeamlessM4T |    10 |  0.7427 |  34.26 |         2.91 |                      3.39 | 0.9821 | 0.2156 |
| retranslation | Sliding Window       | SeamlessM4T |    12 |  0.7500 |  34.58 |         3.46 |                      4.00 | 1.1069 | 0.2452 |
| retranslation | Sliding Window       | SeamlessM4T |    14 |  0.7560 |  34.77 |         4.07 |                      4.70 | 1.2750 | 0.2840 |
| retranslation | VAD + Sliding Window | Canary      |   0.6 |  0.7378 |  33.73 |         2.52 |                      2.77 | 0.1224 | 0.0856 |
| retranslation | VAD + Sliding Window | Canary      |   0.5 |  0.7438 |  34.02 |         2.55 |                      2.82 | 0.1341 | 0.0880 |
| retranslation | VAD + Sliding Window | Canary      |   0.4 |  0.7482 |  34.42 |         2.60 |                      2.88 | 0.1439 | 0.0924 |
| retranslation | VAD + Sliding Window | Canary      |   0.3 |  0.7539 |  34.85 |         2.63 |                      2.90 | 0.1588 | 0.0899 |
| retranslation | VAD + Sliding Window | SeamlessM4T |   0.6 |  0.6941 |  28.70 |         2.55 |                      2.84 | 0.1531 | 0.0978 |
| retranslation | VAD + Sliding Window | SeamlessM4T |   0.5 |  0.6968 |  28.99 |         2.59 |                      2.87 | 0.1610 | 0.0988 |
| retranslation | VAD + Sliding Window | SeamlessM4T |   0.4 |  0.6980 |  29.27 |         2.67 |                      2.97 | 0.1747 | 0.1014 |
| retranslation | VAD + Sliding Window | SeamlessM4T |   0.3 |  0.7038 |  29.88 |         2.72 |                      3.03 | 0.1978 | 0.1038 |
| incremental   | StreamAtt            | SeamlessM4T |     2 |  0.7634 |  34.76 |         1.95 |                      2.23 | 0.0000 | 0.2290 |
| incremental   | StreamAtt            | SeamlessM4T |     4 |  0.7720 |  35.77 |         2.24 |                      2.53 | 0.0000 | 0.2422 |
| incremental   | StreamAtt            | SeamlessM4T |     6 |  0.7744 |  36.38 |         2.53 |                      2.82 | 0.0000 | 0.2522 |
| incremental   | StreamAtt            | SeamlessM4T |     8 |  0.7743 |  36.56 |         2.84 |                      3.15 | 0.0000 | 0.2728 |

## en-it

| Speech P.     | Method               | Model       | w/f/t | COMET ↑ | BLEU ↑ | StreamLAAL ↓ | StreamLAAL<sub>CA</sub> ↓ |   NE ↓ |  RTF ↓ |
| ------------- | -------------------- | ----------- | ----: | ------: | -----: | -----------: | ------------------------: | -----: | -----: |
| retranslation | Sliding Window       | Canary      |     8 |  0.7887 |  27.33 |         2.48 |                      2.80 | 0.5773 | 0.1404 |
| retranslation | Sliding Window       | Canary      |    10 |  0.7921 |  27.24 |         3.05 |                      3.43 | 0.8284 | 0.1654 |
| retranslation | Sliding Window       | Canary      |    12 |  0.7973 |  27.38 |         3.42 |                      3.88 | 0.9569 | 0.1986 |
| retranslation | Sliding Window       | Canary      |    14 |  0.8007 |  28.02 |         3.99 |                      4.50 | 1.1579 | 0.2299 |
| retranslation | Sliding Window       | SeamlessM4T |     8 |  0.7214 |  22.14 |         2.37 |                      2.74 | 0.8585 | 0.1707 |
| retranslation | Sliding Window       | SeamlessM4T |    10 |  0.7353 |  22.99 |         2.87 |                      3.31 | 1.0321 | 0.2011 |
| retranslation | Sliding Window       | SeamlessM4T |    12 |  0.7392 |  23.18 |         3.53 |                      4.06 | 1.1914 | 0.2371 |
| retranslation | Sliding Window       | SeamlessM4T |    14 |  0.7425 |  23.53 |         4.19 |                      4.82 | 1.3329 | 0.2852 |
| retranslation | VAD + Sliding Window | Canary      |   0.6 |  0.7352 |  23.31 |         2.50 |                      2.74 | 0.1307 | 0.0812 |
| retranslation | VAD + Sliding Window | Canary      |   0.5 |  0.7409 |  23.58 |         2.54 |                      2.80 | 0.1373 | 0.0828 |
| retranslation | VAD + Sliding Window | Canary      |   0.4 |  0.7443 |  23.67 |         2.56 |                      2.82 | 0.1460 | 0.0845 |
| retranslation | VAD + Sliding Window | Canary      |   0.3 |  0.7502 |  23.86 |         2.60 |                      2.86 | 0.1616 | 0.0835 |
| retranslation | VAD + Sliding Window | SeamlessM4T |   0.6 |  0.6892 |  19.92 |         2.54 |                      2.82 | 0.1613 | 0.0924 |
| retranslation | VAD + Sliding Window | SeamlessM4T |   0.5 |  0.6924 |  20.14 |         2.57 |                      2.85 | 0.1704 | 0.0928 |
| retranslation | VAD + Sliding Window | SeamlessM4T |   0.4 |  0.6972 |  20.39 |         2.59 |                      2.88 | 0.1845 | 0.0954 |
| retranslation | VAD + Sliding Window | SeamlessM4T |   0.3 |  0.7018 |  20.96 |         2.77 |                      3.08 | 0.2039 | 0.0989 |
| incremental   | StreamAtt            | SeamlessM4T |     2 |  0.7531 |  24.26 |         1.97 |                      2.24 | 0.0000 | 0.2151 |
| incremental   | StreamAtt            | SeamlessM4T |     4 |  0.7576 |  24.95 |         2.26 |                      2.53 | 0.0000 | 0.2270 |
| incremental   | StreamAtt            | SeamlessM4T |     6 |  0.7646 |  25.66 |         2.54 |                      2.83 | 0.0000 | 0.2430 |
| incremental   | StreamAtt            | SeamlessM4T |     8 |  0.7641 |  25.61 |         2.83 |                      3.12 | 0.0000 | 0.2527 |


## en-nl

| Speech P.     | Method               | Model       | w/f/t | COMET ↑ |  BLEU ↑ | StreamLAAL ↓ | StreamLAAL<sub>CA</sub> ↓ |   NE ↓ |  RTF ↓ |
| ------------- | -------------------- | ----------- | ----: | ------: | ------: | -----------: | ------------------------: | -----: | -----: |
| retranslation | Sliding Window       | Canary      |     8 |  0.7898 |   30.78 |         2.56 |                      2.88 | 0.6302 | 0.1384 |
| retranslation | Sliding Window       | Canary      |    10 |  0.7946 |   30.13 |         3.06 |                      3.45 | 0.8527 | 0.1719 |
| retranslation | Sliding Window       | Canary      |    12 |  0.8055 |   30.54 |         3.53 |                      4.00 | 1.0112 | 0.2035 |
| retranslation | Sliding Window       | Canary      |    14 |  0.8089 |   30.82 |         4.04 |                      4.53 | 1.1992 | 0.2153 |
| retranslation | Sliding Window       | SeamlessM4T |     8 |  0.7318 |   26.90 |         2.48 |                      2.86 | 0.8490 | 0.1747 |
| retranslation | Sliding Window       | SeamlessM4T |    10 |  0.7564 |   27.97 |         2.86 |                      3.29 | 1.0153 | 0.1963 |
| retranslation | Sliding Window       | SeamlessM4T |    12 |  0.7631 |   27.99 |         3.50 |                      4.00 | 1.1656 | 0.2283 |
| retranslation | Sliding Window       | SeamlessM4T |    14 |  0.7657 |   27.97 |         4.09 |                      4.67 | 1.3338 | 0.2669 |
| retranslation | VAD + Sliding Window | Canary      |   0.6 |  0.7522 |   26.63 |         2.56 |                      2.80 | 0.1401 | 0.0814 |
| retranslation | VAD + Sliding Window | Canary      |   0.5 |  0.7555 |   26.94 |         2.57 |                      2.81 | 0.1509 | 0.0800 |
| retranslation | VAD + Sliding Window | Canary      |   0.4 |  0.7605 |   27.16 |         2.59 |                      2.84 | 0.1602 | 0.0813 |
| retranslation | VAD + Sliding Window | Canary      |   0.3 |  0.7644 |   27.22 |         2.61 |                      2.88 | 0.1705 | 0.0846 |
| retranslation | VAD + Sliding Window | SeamlessM4T |   0.6 |  0.7148 | 24.0094 |         2.48 |                      2.74 | 0.1599 | 0.0907 |
| retranslation | VAD + Sliding Window | SeamlessM4T |   0.5 |  0.7172 | 24.3518 |         2.55 |                      2.83 | 0.1673 | 0.0921 |
| retranslation | VAD + Sliding Window | SeamlessM4T |   0.4 |  0.7206 | 24.7816 |         2.61 |                      2.89 | 0.1811 | 0.0945 |
| retranslation | VAD + Sliding Window | SeamlessM4T |   0.3 |  0.7262 | 25.2648 |         2.70 |                      3.00 | 0.2038 | 0.0975 |
| incremental   | StreamAtt            | SeamlessM4T |     2 |  0.7775 |   28.70 |         1.86 |                      2.12 | 0.0000 | 0.2142 |
| incremental   | StreamAtt            | SeamlessM4T |     4 |  0.7835 |   29.31 |         2.16 |                      2.43 | 0.0000 | 0.2270 |
| incremental   | StreamAtt            | SeamlessM4T |     6 |  0.7859 |   29.81 |         2.43 |                      2.69 | 0.0000 | 0.2366 |
| incremental   | StreamAtt            | SeamlessM4T |     8 |  0.7849 |   29.98 |         2.74 |                      3.02 | 0.0000 | 0.2547 |


## en-pt

| Speech P.     | Method               | Model       | w/f/t | COMET ↑ | BLEU ↑ | StreamLAAL ↓ | StreamLAAL<sub>CA</sub> ↓ |   NE ↓ |  RTF ↓ |
| ------------- | -------------------- | ----------- | ----: | ------: | -----: | -----------: | ------------------------: | -----: | -----: |
| retranslation | Sliding Window       | Canary      |     8 |  0.8048 |  28.87 |         2.42 |                      2.74 | 0.5113 | 0.1373 |
| retranslation | Sliding Window       | Canary      |    10 |  0.8069 |  28.71 |         2.93 |                      3.38 | 0.7146 | 0.2001 |
| retranslation | Sliding Window       | Canary      |    12 |  0.8098 |  29.02 |         3.22 |                      3.75 | 0.8592 | 0.2281 |
| retranslation | Sliding Window       | Canary      |    14 |  0.8117 |  29.18 |         3.71 |                      4.20 | 1.0196 | 0.2162 |
| retranslation | Sliding Window       | SeamlessM4T |     8 |  0.7547 |  28.37 |         2.45 |                      2.82 | 0.8896 | 0.1687 |
| retranslation | Sliding Window       | SeamlessM4T |    10 |  0.7716 |  29.43 |         2.90 |                      3.35 | 1.0518 | 0.2029 |
| retranslation | Sliding Window       | SeamlessM4T |    12 |  0.7784 |  29.75 |         3.56 |                      4.07 | 1.1951 | 0.2335 |
| retranslation | Sliding Window       | SeamlessM4T |    14 |  0.7828 |  29.68 |         4.28 |                      4.88 | 1.3733 | 0.2653 |
| retranslation | VAD + Sliding Window | Canary      |   0.6 |  0.7576 |  24.52 |         2.60 |                      2.84 | 0.1210 | 0.0825 |
| retranslation | VAD + Sliding Window | Canary      |   0.5 |  0.7625 |  24.81 |         2.62 |                      2.87 | 0.1278 | 0.0819 |
| retranslation | VAD + Sliding Window | Canary      |   0.4 |  0.7657 |  25.07 |         2.64 |                      2.89 | 0.1391 | 0.0841 |
| retranslation | VAD + Sliding Window | Canary      |   0.3 |  0.7711 |  25.54 |         2.67 |                      2.94 | 0.1531 | 0.0857 |
| retranslation | VAD + Sliding Window | SeamlessM4T |   0.6 |  0.7270 |  25.30 |         2.46 |                      2.72 | 0.1581 | 0.0905 |
| retranslation | VAD + Sliding Window | SeamlessM4T |   0.5 |  0.7282 |  25.62 |         2.52 |                      2.79 | 0.1677 | 0.0916 |
| retranslation | VAD + Sliding Window | SeamlessM4T |   0.4 |  0.7314 |  25.99 |         2.58 |                      2.86 | 0.1794 | 0.0951 |
| retranslation | VAD + Sliding Window | SeamlessM4T |   0.3 |  0.7403 |  26.35 |         2.69 |                      2.99 | 0.2031 | 0.0989 |
| incremental   | StreamAtt            | SeamlessM4T |     2 |  0.7867 |  29.57 |         1.93 |                      2.19 | 0.0000 | 0.2114 |
| incremental   | StreamAtt            | SeamlessM4T |     4 |  0.7915 |  30.68 |         2.23 |                      2.49 | 0.0000 | 0.2221 |
| incremental   | StreamAtt            | SeamlessM4T |     6 |  0.7949 |  31.20 |         2.52 |                      2.79 | 0.0000 | 0.2327 |
| incremental   | StreamAtt            | SeamlessM4T |     8 |  0.7952 |  31.35 |         2.79 |                      3.07 | 0.0000 | 0.2468 |

## en-ro

| Speech P.     | Method               | Model       | w/f/t | COMET ↑ | BLEU ↑ | StreamLAAL ↓ | StreamLAAL<sub>CA</sub> ↓ |   NE ↓ |  RTF ↓ |
| ------------- | -------------------- | ----------- | ----: | ------: | -----: | -----------: | ------------------------: | -----: | -----: |
| retranslation | Sliding Window       | Canary      |     8 |  0.8071 |  25.14 |         2.47 |                      2.82 | 0.5655 | 0.1459 |
| retranslation | Sliding Window       | Canary      |    10 |  0.8092 |  24.87 |         2.98 |                      3.38 | 0.8125 | 0.1714 |
| retranslation | Sliding Window       | Canary      |    12 |  0.8162 |  25.48 |         3.39 |                      3.85 | 0.9404 | 0.2029 |
| retranslation | Sliding Window       | Canary      |    14 |  0.8192 |  25.56 |         3.91 |                      4.43 | 1.1203 | 0.2322 |
| retranslation | Sliding Window       | SeamlessM4T |     8 |  0.7300 |  21.45 |         2.47 |                      2.85 | 0.8613 | 0.1760 |
| retranslation | Sliding Window       | SeamlessM4T |    10 |  0.7470 |  22.16 |         3.08 |                      3.55 | 1.0327 | 0.2108 |
| retranslation | Sliding Window       | SeamlessM4T |    12 |  0.7542 |  22.21 |         3.72 |                      4.26 | 1.2135 | 0.2500 |
| retranslation | Sliding Window       | SeamlessM4T |    14 |  0.7534 |  21.84 |         4.54 |                      5.17 | 1.3673 | 0.2833 |
| retranslation | VAD + Sliding Window | Canary      |   0.6 |  0.7531 |  21.51 |         2.59 |                      2.83 | 0.1312 | 0.0818 |
| retranslation | VAD + Sliding Window | Canary      |   0.5 |  0.7581 |  21.87 |         2.58 |                      2.87 | 0.1411 | 0.0824 |
| retranslation | VAD + Sliding Window | Canary      |   0.4 |  0.7622 |  21.94 |         2.51 |                      2.76 | 0.1484 | 0.0830 |
| retranslation | VAD + Sliding Window | Canary      |   0.3 |  0.7699 |  22.37 |         2.62 |                      2.87 | 0.1600 | 0.0842 |
| retranslation | VAD + Sliding Window | SeamlessM4T |   0.6 |  0.7056 |  19.67 |         2.58 |                      2.87 | 0.1662 | 0.0973 |
| retranslation | VAD + Sliding Window | SeamlessM4T |   0.5 |  0.7063 |  19.61 |         2.60 |                      2.88 | 0.1718 | 0.0967 |
| retranslation | VAD + Sliding Window | SeamlessM4T |   0.4 |  0.7113 |  19.89 |         2.67 |                      2.97 | 0.1856 | 0.1007 |
| retranslation | VAD + Sliding Window | SeamlessM4T |   0.3 |  0.7146 |  20.24 |         2.75 |                      3.06 | 0.2038 | 0.1024 |
| incremental   | StreamAtt            | SeamlessM4T |     2 |  0.7702 |  23.47 |         1.95 |                      2.23 | 0.0000 | 0.2167 |
| incremental   | StreamAtt            | SeamlessM4T |     4 |  0.7774 |  24.17 |         2.28 |                      2.57 | 0.0000 | 0.2358 |
| incremental   | StreamAtt            | SeamlessM4T |     6 |  0.7807 |  24.75 |         2.60 |                      2.90 | 0.0000 | 0.2472 |
| incremental   | StreamAtt            | SeamlessM4T |     8 |  0.7827 |  24.75 |         2.84 |                      3.14 | 0.0000 | 0.2578 |


## en-ru

| Speech P.     | Method               | Model       | w/f/t | COMET ↑ | BLEU ↑ | StreamLAAL ↓ | StreamLAAL<sub>CA</sub> ↓ |   NE ↓ |  RTF ↓ |
| ------------- | -------------------- | ----------- | ----: | ------: | -----: | -----------: | ------------------------: | -----: | -----: |
| retranslation | Sliding Window       | Canary      |     8 |  0.7481 |  18.82 |         2.48 |                      2.82 | 0.5988 | 0.1536 |
| retranslation | Sliding Window       | Canary      |    10 |  0.7513 |  18.63 |         3.06 |                      3.47 | 0.8182 | 0.1778 |
| retranslation | Sliding Window       | Canary      |    12 |  0.7625 |  19.22 |         3.52 |                      3.99 | 0.9862 | 0.2117 |
| retranslation | Sliding Window       | Canary      |    14 |  0.7631 |  19.09 |         3.98 |                      4.52 | 1.1400 | 0.2427 |
| retranslation | Sliding Window       | SeamlessM4T |     8 |  0.6689 |  14.88 |         2.45 |                      2.82 | 0.8318 | 0.1725 |
| retranslation | Sliding Window       | SeamlessM4T |    10 |  0.6863 |  15.35 |         2.97 |                      3.42 | 1.0134 | 0.2126 |
| retranslation | Sliding Window       | SeamlessM4T |    12 |  0.6959 |  15.70 |         3.51 |                      4.04 | 1.1643 | 0.2434 |
| retranslation | Sliding Window       | SeamlessM4T |    14 |  0.6986 |  15.60 |         4.54 |                      5.15 | 1.3376 | 0.2677 |
| retranslation | VAD + Sliding Window | Canary      |   0.6 |  0.7054 |  15.93 |         2.46 |                      2.71 | 0.1425 | 0.0845 |
| retranslation | VAD + Sliding Window | Canary      |   0.5 |  0.7100 |  16.10 |         2.50 |                      2.77 | 0.1507 | 0.0880 |
| retranslation | VAD + Sliding Window | Canary      |   0.4 |  0.7116 |  16.28 |         2.43 |                      2.70 | 0.1551 | 0.0915 |
| retranslation | VAD + Sliding Window | Canary      |   0.3 |  0.7183 |  16.52 |         2.47 |                      2.75 | 0.1714 | 0.0920 |
| retranslation | VAD + Sliding Window | SeamlessM4T |   0.6 |  0.6514 |  13.32 |         2.64 |                      2.91 | 0.1682 | 0.0943 |
| retranslation | VAD + Sliding Window | SeamlessM4T |   0.5 |  0.6535 |  13.46 |         2.58 |                      2.86 | 0.1769 | 0.0968 |
| retranslation | VAD + Sliding Window | SeamlessM4T |   0.4 |  0.6579 |  13.75 |         2.64 |                      2.93 | 0.1907 | 0.0991 |
| retranslation | VAD + Sliding Window | SeamlessM4T |   0.3 |  0.6632 |  14.12 |         2.89 |                      3.19 | 0.2139 | 0.1023 |
| incremental   | StreamAtt            | SeamlessM4T |     2 |  0.7243 |  16.83 |         2.28 |                      2.58 | 0.0000 | 0.2409 |
| incremental   | StreamAtt            | SeamlessM4T |     4 |  0.7282 |  17.31 |         2.50 |                      2.81 | 0.0000 | 0.2468 |
| incremental   | StreamAtt            | SeamlessM4T |     6 |  0.7340 |  17.64 |         2.91 |                      3.24 | 0.0000 | 0.2580 |
| incremental   | StreamAtt            | SeamlessM4T |     8 |  0.7354 |  17.79 |         3.12 |                      3.45 | 0.0000 | 0.2708 |
