

[toc]



# Question

* **Filename typo**: "salat" ---> "salad"

# Results

***

## 1. loss='categorical_crossentropy', optimizer='adam', activation="softmax"

### Without activity

***

| Input: WO_Activity    | Input(%):  10.0 %     |                       |                       |                        |
| --------------------- | --------------------- | --------------------- | --------------------- | ---------------------- |
| timestep  1 : 0.10066 | timestep  2 : 0.20031 | timestep  3 : 0.16908 | timestep  4 : 0.21369 | timestep  5 : 0.16411  |
| timestep  6 : 0.33992 | timestep  7 : 0.36277 | timestep  8 : 0.25333 | timestep  9 : 0.26786 | timestep  10 : 0.19167 |
| timestep  11 : 0.375  | timestep  12 : 0.4375 | timestep  13 : 0.15   | timestep  14 : 0.0    | timestep  15 : 0.0     |
| timestep  16 : 0.0    | timestep  17 : 0.0    | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0     |
| timestep  21 : 0.0    | timestep  22 : 0.25   | timestep  23 : 0.0    |                       |                        |

| Input: WO_Activity     | Input(%):  20.0 %     |                       |                       |                        |
| ---------------------- | --------------------- | --------------------- | --------------------- | ---------------------- |
| timestep  1 : 0.27499  | timestep  2 : 0.29275 | timestep  3 : 0.2738  | timestep  4 : 0.27137 | timestep  5 : 0.40463  |
| timestep  6 : 0.38869  | timestep  7 : 0.33201 | timestep  8 : 0.35614 | timestep  9 : 0.07692 | timestep  10 : 0.36667 |
| timestep  11 : 0.16667 | timestep  12 : 0.25   | timestep  13 : 0.0    | timestep  14 : 0.25   | timestep  15 : 0.0     |
| timestep  16 : 0.0     | timestep  17 : 0.25   | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0     |
| timestep  21 : 0.0     | timestep  22 : 0.0    | timestep  23 : 0.0    |                       |                        |

| Input: WO_Activity    | Input(%):  30.0 %     |                       |                       |                       |
| --------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| timestep  1 : 0.38085 | timestep  2 : 0.34042 | timestep  3 : 0.29426 | timestep  4 : 0.45129 | timestep  5 : 0.40229 |
| timestep  6 : 0.40893 | timestep  7 : 0.29618 | timestep  8 : 0.20833 | timestep  9 : 0.19444 | timestep  10 : 0.475  |
| timestep  11 : 0.625  | timestep  12 : 0.0    | timestep  13 : 0.25   | timestep  14 : 0.0    | timestep  15 : 0.0    |
| timestep  16 : 0.0    | timestep  17 : 0.0    | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0    |
| timestep  21 : 0.0    | timestep  22 : 0.0    | timestep  23 : 0.0    |                       |                       |

| Input: WO_Activity    | Input(%):  40.0 %     |                       |                       |                       |
| --------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| timestep  1 : 0.49445 | timestep  2 : 0.37644 | timestep  3 : 0.41375 | timestep  4 : 0.46221 | timestep  5 : 0.36411 |
| timestep  6 : 0.40793 | timestep  7 : 0.46843 | timestep  8 : 0.26111 | timestep  9 : 0.125   | timestep  10 : 0.275  |
| timestep  11 : 0.25   | timestep  12 : 0.0    | timestep  13 : 0.0    | timestep  14 : 0.0    | timestep  15 : 0.0    |
| timestep  16 : 0.0    | timestep  17 : 0.0    | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0    |
| timestep  21 : 0.0    | timestep  22 : 0.0    | timestep  23 : 0.0    |                       |                       |

| Input: WO_Activity    | Input(%):  50.0 %     |                       |                       |                       |
| --------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| timestep  1 : 0.45393 | timestep  2 : 0.47437 | timestep  3 : 0.45315 | timestep  4 : 0.51937 | timestep  5 : 0.38557 |
| timestep  6 : 0.47917 | timestep  7 : 0.5     | timestep  8 : 0.36111 | timestep  9 : 0.125   | timestep  10 : 0.25   |
| timestep  11 : 0.25   | timestep  12 : 0.0    | timestep  13 : 0.0    | timestep  14 : 0.0    | timestep  15 : 0.0    |
| timestep  16 : 0.0    | timestep  17 : 0.0    | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0    |
| timestep  21 : 0.0    | timestep  22 : 0.0    | timestep  23 : 0.0    |                       |                       |



### With activity

***

| Input: W_Activity     | Input(%):  10.0 %     |                        |                       |                       |
| --------------------- | --------------------- | ---------------------- | --------------------- | --------------------- |
| timestep  1 : 0.45677 | timestep  2 : 0.47518 | timestep  3 : 0.45275  | timestep  4 : 0.50401 | timestep  5 : 0.57158 |
| timestep  6 : 0.52093 | timestep  7 : 0.65255 | timestep  8 : 0.49424  | timestep  9 : 0.50298 | timestep  10 : 0.3631 |
| timestep  11 : 0.65   | timestep  12 : 0.6875 | timestep  13 : 0.56667 | timestep  14 : 0.0    | timestep  15 : 0.25   |
| timestep  16 : 0.0    | timestep  17 : 0.0    | timestep  18 : 0.25    | timestep  19 : 0.0    | timestep  20 : 0.0    |
| timestep  21 : 0.25   | timestep  22 : 0.25   | timestep  23 : 0.0     |                       |                       |

| Input: W_Activity     | Input(%):  20.0 %     |                        |                       |                       |
| --------------------- | --------------------- | ---------------------- | --------------------- | --------------------- |
| timestep  1 : 0.47876 | timestep  2 : 0.56698 | timestep  3 : 0.45736  | timestep  4 : 0.56825 | timestep  5 : 0.63454 |
| timestep  6 : 0.57921 | timestep  7 : 0.66265 | timestep  8 : 0.49264  | timestep  9 : 0.47051 | timestep  10 : 0.525  |
| timestep  11 : 0.5    | timestep  12 : 0.75   | timestep  13 : 0.41667 | timestep  14 : 0.0    | timestep  15 : 0.0    |
| timestep  16 : 0.25   | timestep  17 : 0.25   | timestep  18 : 0.0     | timestep  19 : 0.0    | timestep  20 : 0.0    |
| timestep  21 : 0.0    | timestep  22 : 0.0    | timestep  23 : 0.0     |                       |                       |

| Input: W_Activity     | Input(%):  30.0 %      |                       |                       |                       |
| --------------------- | ---------------------- | --------------------- | --------------------- | --------------------- |
| timestep  1 : 0.56126 | timestep  2 : 0.51253  | timestep  3 : 0.45243 | timestep  4 : 0.57758 | timestep  5 : 0.55696 |
| timestep  6 : 0.53275 | timestep  7 : 0.77557  | timestep  8 : 0.45437 | timestep  9 : 0.53056 | timestep  10 : 0.2875 |
| timestep  11 : 0.625  | timestep  12 : 0.16667 | timestep  13 : 0.5    | timestep  14 : 0.0    | timestep  15 : 0.0    |
| timestep  16 : 0.0    | timestep  17 : 0.0     | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0    |
| timestep  21 : 0.0    | timestep  22 : 0.0     | timestep  23 : 0.0    |                       |                       |

| Input: W_Activity     | Input(%):  40.0 %     |                       |                       |                       |
| --------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| timestep  1 : 0.57909 | timestep  2 : 0.52768 | timestep  3 : 0.49848 | timestep  4 : 0.58245 | timestep  5 : 0.60421 |
| timestep  6 : 0.47669 | timestep  7 : 0.82955 | timestep  8 : 0.32143 | timestep  9 : 0.4375  | timestep  10 : 0.625  |
| timestep  11 : 0.5    | timestep  12 : 0.25   | timestep  13 : 0.0    | timestep  14 : 0.0    | timestep  15 : 0.0    |
| timestep  16 : 0.0    | timestep  17 : 0.0    | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0    |
| timestep  21 : 0.0    | timestep  22 : 0.0    | timestep  23 : 0.0    |                       |                       |

| Input: W_Activity     | Input(%):  50.0 %     |                       |                       |                       |
| --------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| timestep  1 : 0.53786 | timestep  2 : 0.56607 | timestep  3 : 0.57247 | timestep  4 : 0.61809 | timestep  5 : 0.48816 |
| timestep  6 : 0.53472 | timestep  7 : 0.61389 | timestep  8 : 0.42361 | timestep  9 : 0.1875  | timestep  10 : 0.5    |
| timestep  11 : 0.25   | timestep  12 : 0.25   | timestep  13 : 0.0    | timestep  14 : 0.0    | timestep  15 : 0.0    |
| timestep  16 : 0.0    | timestep  17 : 0.0    | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0    |
| timestep  21 : 0.0    | timestep  22 : 0.0    | timestep  23 : 0.0    |                       |                       |





## 2. loss='categorical_crossentropy', optimizer='adam', activation="relu"

### Without activity

***

Almost all timesteps with accuracy 0.

### With activity

***

| Input: W_Activity     | Input(%):  10.0 %     |                       |                       |                       |
| --------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| timestep  1 : 0.36458 | timestep  2 : 0.41333 | timestep  3 : 0.35187 | timestep  4 : 0.37277 | timestep  5 : 0.36898 |
| timestep  6 : 0.32369 | timestep  7 : 0.33706 | timestep  8 : 0.41396 | timestep  9 : 0.23661 | timestep  10 : 0.5381 |
| timestep  11 : 0.425  | timestep  12 : 0.25   | timestep  13 : 0.5    | timestep  14 : 0.25   | timestep  15 : 0.25   |
| timestep  16 : 0.0    | timestep  17 : 0.0    | timestep  18 : 0.25   | timestep  19 : 0.0    | timestep  20 : 0.0    |
| timestep  21 : 0.0    | timestep  22 : 0.25   | timestep  23 : 0.0    |                       |                       |

| Input: W_Activity      | Input(%):  20.0 %     |                       |                       |                       |
| ---------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| timestep  1 : 0.26538  | timestep  2 : 0.29427 | timestep  3 : 0.24705 | timestep  4 : 0.27049 | timestep  5 : 0.32965 |
| timestep  6 : 0.5335   | timestep  7 : 0.26786 | timestep  8 : 0.69348 | timestep  9 : 0.17692 | timestep  10 : 0.625  |
| timestep  11 : 0.29167 | timestep  12 : 0.5    | timestep  13 : 0.25   | timestep  14 : 0.5    | timestep  15 : 0.0    |
| timestep  16 : 0.0     | timestep  17 : 0.25   | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0    |
| timestep  21 : 0.0     | timestep  22 : 0.0    | timestep  23 : 0.0    |                       |                       |

| Input: W_Activity     | Input(%):  30.0 %      |                       |                       |                       |
| --------------------- | ---------------------- | --------------------- | --------------------- | --------------------- |
| timestep  1 : 0.47893 | timestep  2 : 0.43075  | timestep  3 : 0.33486 | timestep  4 : 0.2948  | timestep  5 : 0.4031  |
| timestep  6 : 0.30963 | timestep  7 : 0.66619  | timestep  8 : 0.4127  | timestep  9 : 0.57222 | timestep  10 : 0.2875 |
| timestep  11 : 0.625  | timestep  12 : 0.16667 | timestep  13 : 0.5    | timestep  14 : 0.0    | timestep  15 : 0.0    |
| timestep  16 : 0.0    | timestep  17 : 0.0     | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0    |
| timestep  21 : 0.0    | timestep  22 : 0.0     | timestep  23 : 0.0    |                       |                       |

| Input: W_Activity     | Input(%):  40.0 %     |                       |                       |                      |
| --------------------- | --------------------- | --------------------- | --------------------- | -------------------- |
| timestep  1 : 0.05081 | timestep  2 : 0.18615 | timestep  3 : 0.21607 | timestep  4 : 0.34378 | timestep  5 : 0.1604 |
| timestep  6 : 0.07798 | timestep  7 : 0.06818 | timestep  8 : 0.0     | timestep  9 : 0.0     | timestep  10 : 0.0   |
| timestep  11 : 0.0    | timestep  12 : 0.0    | timestep  13 : 0.0    | timestep  14 : 0.0    | timestep  15 : 0.0   |
| timestep  16 : 0.0    | timestep  17 : 0.0    | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0   |
| timestep  21 : 0.0    | timestep  22 : 0.0    | timestep  23 : 0.0    |                       |                      |

| Input: W_Activity     | Input(%):  50.0 %     |                       |                       |                       |
| --------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| timestep  1 : 0.3696  | timestep  2 : 0.42661 | timestep  3 : 0.36937 | timestep  4 : 0.40473 | timestep  5 : 0.40699 |
| timestep  6 : 0.35417 | timestep  7 : 0.475   | timestep  8 : 0.3125  | timestep  9 : 0.1875  | timestep  10 : 0.5    |
| timestep  11 : 0.0    | timestep  12 : 0.25   | timestep  13 : 0.0    | timestep  14 : 0.0    | timestep  15 : 0.0    |
| timestep  16 : 0.0    | timestep  17 : 0.0    | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0    |
| timestep  21 : 0.0    | timestep  22 : 0.0    | timestep  23 : 0.0    |                       |                       |



## 3. loss='mse', optimizer='adam', activation="relu"

### Without activity

***

| Input: WO_Activity    | Input(%):  10.0 %     |                       |                       |                     |
| --------------------- | --------------------- | --------------------- | --------------------- | ------------------- |
| timestep  1 : 0.08982 | timestep  2 : 0.07703 | timestep  3 : 0.08787 | timestep  4 : 0.11487 | timestep  5 : 0.117 |
| timestep  6 : 0.15736 | timestep  7 : 0.23984 | timestep  8 : 0.1336  | timestep  9 : 0.12649 | timestep  10 : 0.0  |
| timestep  11 : 0.0    | timestep  12 : 0.0    | timestep  13 : 0.0    | timestep  14 : 0.0    | timestep  15 : 0.0  |
| timestep  16 : 0.0    | timestep  17 : 0.0    | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0  |
| timestep  21 : 0.0    | timestep  22 : 0.0    | timestep  23 : 0.0    |                       |                     |

| Input: WO_Activity    | Input(%):  20.0 %     |                       |                       |                       |
| --------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| timestep  1 : 0.12231 | timestep  2 : 0.13563 | timestep  3 : 0.1427  | timestep  4 : 0.17238 | timestep  5 : 0.26205 |
| timestep  6 : 0.21297 | timestep  7 : 0.14624 | timestep  8 : 0.12905 | timestep  9 : 0.0     | timestep  10 : 0.1    |
| timestep  11 : 0.0    | timestep  12 : 0.0    | timestep  13 : 0.0    | timestep  14 : 0.0    | timestep  15 : 0.0    |
| timestep  16 : 0.0    | timestep  17 : 0.0    | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0    |
| timestep  21 : 0.0    | timestep  22 : 0.0    | timestep  23 : 0.0    |                       |                       |

| Input: WO_Activity    | Input(%):  30.0 %     |                       |                      |                       |
| --------------------- | --------------------- | --------------------- | -------------------- | --------------------- |
| timestep  1 : 0.30551 | timestep  2 : 0.25608 | timestep  3 : 0.30541 | timestep  4 : 0.3916 | timestep  5 : 0.36997 |
| timestep  6 : 0.34903 | timestep  7 : 0.31736 | timestep  8 : 0.20833 | timestep  9 : 0.125  | timestep  10 : 0.15   |
| timestep  11 : 0.25   | timestep  12 : 0.0    | timestep  13 : 0.25   | timestep  14 : 0.0   | timestep  15 : 0.0    |
| timestep  16 : 0.0    | timestep  17 : 0.0    | timestep  18 : 0.0    | timestep  19 : 0.0   | timestep  20 : 0.0    |
| timestep  21 : 0.0    | timestep  22 : 0.0    | timestep  23 : 0.0    |                      |                       |

| Input: WO_Activity    | Input(%):  40.0 %     |                       |                       |                       |
| --------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| timestep  1 : 0.38637 | timestep  2 : 0.30762 | timestep  3 : 0.36046 | timestep  4 : 0.46856 | timestep  5 : 0.38057 |
| timestep  6 : 0.29107 | timestep  7 : 0.42503 | timestep  8 : 0.15    | timestep  9 : 0.0     | timestep  10 : 0.525  |
| timestep  11 : 0.25   | timestep  12 : 0.0    | timestep  13 : 0.0    | timestep  14 : 0.0    | timestep  15 : 0.0    |
| timestep  16 : 0.0    | timestep  17 : 0.0    | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0    |
| timestep  21 : 0.0    | timestep  22 : 0.0    | timestep  23 : 0.0    |                       |                       |

| Input: WO_Activity    | Input(%):  50.0 %     |                       |                       |                       |
| --------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| timestep  1 : 0.43577 | timestep  2 : 0.46328 | timestep  3 : 0.55876 | timestep  4 : 0.52501 | timestep  5 : 0.40361 |
| timestep  6 : 0.49306 | timestep  7 : 0.125   | timestep  8 : 0.375   | timestep  9 : 0.375   | timestep  10 : 0.5    |
| timestep  11 : 0.0    | timestep  12 : 0.0    | timestep  13 : 0.0    | timestep  14 : 0.0    | timestep  15 : 0.0    |
| timestep  16 : 0.0    | timestep  17 : 0.0    | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0    |
| timestep  21 : 0.0    | timestep  22 : 0.0    | timestep  23 : 0.0    |                       |                       |



### With Activity

***

| Input: W_Activity     | Input(%):  10.0 %     |                       |                       |                       |
| --------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| timestep  1 : 0.23392 | timestep  2 : 0.28737 | timestep  3 : 0.25559 | timestep  4 : 0.35652 | timestep  5 : 0.25882 |
| timestep  6 : 0.29559 | timestep  7 : 0.29253 | timestep  8 : 0.21914 | timestep  9 : 0.20387 | timestep  10 : 0.025  |
| timestep  11 : 0.1625 | timestep  12 : 0.1875 | timestep  13 : 0.15   | timestep  14 : 0.0    | timestep  15 : 0.0    |
| timestep  16 : 0.0    | timestep  17 : 0.25   | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.25   |
| timestep  21 : 0.0    | timestep  22 : 0.0    | timestep  23 : 0.0    |                       |                       |

| Input: W_Activity      | Input(%):  20.0 %     |                       |                       |                        |
| ---------------------- | --------------------- | --------------------- | --------------------- | ---------------------- |
| timestep  1 : 0.08641  | timestep  2 : 0.16987 | timestep  3 : 0.21801 | timestep  4 : 0.32693 | timestep  5 : 0.35114  |
| timestep  6 : 0.26335  | timestep  7 : 0.49977 | timestep  8 : 0.41754 | timestep  9 : 0.36026 | timestep  10 : 0.65417 |
| timestep  11 : 0.79167 | timestep  12 : 0.0    | timestep  13 : 0.25   | timestep  14 : 0.25   | timestep  15 : 0.0     |
| timestep  16 : 0.0     | timestep  17 : 0.25   | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0     |
| timestep  21 : 0.0     | timestep  22 : 0.0    | timestep  23 : 0.0    |                       |                        |

| Input: W_Activity     | Input(%):  30.0 %      |                       |                       |                       |
| --------------------- | ---------------------- | --------------------- | --------------------- | --------------------- |
| timestep  1 : 0.09206 | timestep  2 : 0.17086  | timestep  3 : 0.16057 | timestep  4 : 0.30539 | timestep  5 : 0.3757  |
| timestep  6 : 0.32868 | timestep  7 : 0.79972  | timestep  8 : 0.25198 | timestep  9 : 0.58056 | timestep  10 : 0.4125 |
| timestep  11 : 0.625  | timestep  12 : 0.16667 | timestep  13 : 0.5    | timestep  14 : 0.0    | timestep  15 : 0.0    |
| timestep  16 : 0.0    | timestep  17 : 0.0     | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0    |
| timestep  21 : 0.0    | timestep  22 : 0.0     | timestep  23 : 0.0    |                       |                       |

| Input: W_Activity     | Input(%):  40.0 %     |                       |                       |                       |
| --------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| timestep  1 : 0.1406  | timestep  2 : 0.18442 | timestep  3 : 0.33438 | timestep  4 : 0.36802 | timestep  5 : 0.49444 |
| timestep  6 : 0.39305 | timestep  7 : 0.86222 | timestep  8 : 0.32143 | timestep  9 : 0.4375  | timestep  10 : 0.625  |
| timestep  11 : 0.5    | timestep  12 : 0.25   | timestep  13 : 0.0    | timestep  14 : 0.0    | timestep  15 : 0.0    |
| timestep  16 : 0.0    | timestep  17 : 0.0    | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0    |
| timestep  21 : 0.0    | timestep  22 : 0.0    | timestep  23 : 0.0    |                       |                       |

| Input: W_Activity     | Input(%):  50.0 %     |                       |                       |                       |
| --------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| timestep  1 : 0.12174 | timestep  2 : 0.13969 | timestep  3 : 0.1164  | timestep  4 : 0.31296 | timestep  5 : 0.48965 |
| timestep  6 : 0.53968 | timestep  7 : 0.65    | timestep  8 : 0.54861 | timestep  9 : 0.3125  | timestep  10 : 0.75   |
| timestep  11 : 0.25   | timestep  12 : 0.25   | timestep  13 : 0.0    | timestep  14 : 0.0    | timestep  15 : 0.0    |
| timestep  16 : 0.0    | timestep  17 : 0.0    | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0    |
| timestep  21 : 0.0    | timestep  22 : 0.0    | timestep  23 : 0.0    |                       |                       |







# TCN

***

| Input: WO_Activity    | Input(%):  10.0 %      |                       |                       |                        |
| --------------------- | ---------------------- | --------------------- | --------------------- | ---------------------- |
| timestep  1 : 0.18418 | timestep  2 : 0.23737  | timestep  3 : 0.22124 | timestep  4 : 0.2862  | timestep  5 : 0.31636  |
| timestep  6 : 0.47659 | timestep  7 : 0.49659  | timestep  8 : 0.44988 | timestep  9 : 0.32738 | timestep  10 : 0.44167 |
| timestep  11 : 0.45   | timestep  12 : 0.34375 | timestep  13 : 0.4    | timestep  14 : 0.0    | timestep  15 : 0.0     |
| timestep  16 : 0.0    | timestep  17 : 0.0     | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0     |
| timestep  21 : 0.0    | timestep  22 : 0.25    | timestep  23 : 0.0    |                       |                        |

| Input: WO_Activity     | Input(%):  20.0 %     |                        |                       |                        |
| ---------------------- | --------------------- | ---------------------- | --------------------- | ---------------------- |
| timestep  1 : 0.34712  | timestep  2 : 0.34137 | timestep  3 : 0.31324  | timestep  4 : 0.35773 | timestep  5 : 0.54165  |
| timestep  6 : 0.47569  | timestep  7 : 0.42398 | timestep  8 : 0.48509  | timestep  9 : 0.32308 | timestep  10 : 0.56667 |
| timestep  11 : 0.41667 | timestep  12 : 0.25   | timestep  13 : 0.41667 | timestep  14 : 0.5    | timestep  15 : 0.0     |
| timestep  16 : 0.25    | timestep  17 : 0.25   | timestep  18 : 0.0     | timestep  19 : 0.0    | timestep  20 : 0.0     |
| timestep  21 : 0.0     | timestep  22 : 0.0    | timestep  23 : 0.0     |                       |                        |

| Input: WO_Activity     | Input(%):  30.0 %      |                       |                       |                       |
| ---------------------- | ---------------------- | --------------------- | --------------------- | --------------------- |
| timestep  1 : 0.53642  | timestep  2 : 0.40336  | timestep  3 : 0.37157 | timestep  4 : 0.4663  | timestep  5 : 0.52191 |
| timestep  6 : 0.44746  | timestep  7 : 0.53567  | timestep  8 : 0.49405 | timestep  9 : 0.29444 | timestep  10 : 0.5375 |
| timestep  11 : 0.91667 | timestep  12 : 0.41667 | timestep  13 : 0.25   | timestep  14 : 0.0    | timestep  15 : 0.0    |
| timestep  16 : 0.0     | timestep  17 : 0.0     | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0    |
| timestep  21 : 0.0     | timestep  22 : 0.0     | timestep  23 : 0.0    |                       |                       |

| Input: WO_Activity    | Input(%):  40.0 %     |                       |                       |                       |
| --------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| timestep  1 : 0.53685 | timestep  2 : 0.48398 | timestep  3 : 0.4813  | timestep  4 : 0.47178 | timestep  5 : 0.44262 |
| timestep  6 : 0.39727 | timestep  7 : 0.44318 | timestep  8 : 0.34643 | timestep  9 : 0.4375  | timestep  10 : 0.625  |
| timestep  11 : 0.25   | timestep  12 : 0.25   | timestep  13 : 0.0    | timestep  14 : 0.0    | timestep  15 : 0.0    |
| timestep  16 : 0.0    | timestep  17 : 0.0    | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0    |
| timestep  21 : 0.0    | timestep  22 : 0.0    | timestep  23 : 0.0    |                       |                       |

| Input: WO_Activity    | Input(%):  50.0 %     |                       |                       |                       |
| --------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| timestep  1 : 0.63546 | timestep  2 : 0.60556 | timestep  3 : 0.57954 | timestep  4 : 0.54322 | timestep  5 : 0.54182 |
| timestep  6 : 0.52083 | timestep  7 : 0.76389 | timestep  8 : 0.36111 | timestep  9 : 0.5625  | timestep  10 : 0.5    |
| timestep  11 : 0.25   | timestep  12 : 0.25   | timestep  13 : 0.0    | timestep  14 : 0.0    | timestep  15 : 0.0    |
| timestep  16 : 0.0    | timestep  17 : 0.0    | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0    |
| timestep  21 : 0.0    | timestep  22 : 0.0    | timestep  23 : 0.0    |                       |                       |