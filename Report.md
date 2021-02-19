

[toc]



# Question

* **Filename typo**: "salat" ---> "salad"

# Results

## LSTM

***

## 1. loss='categorical_crossentropy', optimizer='adam', activation="softmax"

### Without activity

***

| Input: WO_Activity    | Input(%):  10.0 %      |                       |                       |                        |
| --------------------- | ---------------------- | --------------------- | --------------------- | ---------------------- |
| timestep  1 : 0.17968 | timestep  2 : 0.24484  | timestep  3 : 0.26585 | timestep  4 : 0.33787 | timestep  5 : 0.35914  |
| timestep  6 : 0.58526 | timestep  7 : 0.45913  | timestep  8 : 0.37256 | timestep  9 : 0.43155 | timestep  10 : 0.44167 |
| timestep  11 : 0.575  | timestep  12 : 0.34375 | timestep  13 : 0.4    | timestep  14 : 0.25   | timestep  15 : 0.0     |
| timestep  16 : 0.0    | timestep  17 : 0.25    | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.25    |
| timestep  21 : 0.0    | timestep  22 : 0.25    | timestep  23 : 0.0    |                       |                        |

| Input: WO_Activity     | Input(%):  20.0 %     |                        |                       |                        |
| ---------------------- | --------------------- | ---------------------- | --------------------- | ---------------------- |
| timestep  1 : 0.35689  | timestep  2 : 0.38423 | timestep  3 : 0.3517   | timestep  4 : 0.42916 | timestep  5 : 0.53648  |
| timestep  6 : 0.56727  | timestep  7 : 0.36308 | timestep  8 : 0.47062  | timestep  9 : 0.43419 | timestep  10 : 0.51667 |
| timestep  11 : 0.47917 | timestep  12 : 0.5    | timestep  13 : 0.66667 | timestep  14 : 0.5    | timestep  15 : 0.25    |
| timestep  16 : 0.25    | timestep  17 : 0.25   | timestep  18 : 0.0     | timestep  19 : 0.0    | timestep  20 : 0.0     |
| timestep  21 : 0.0     | timestep  22 : 0.0    | timestep  23 : 0.0     |                       |                        |

| Input: WO_Activity    | Input(%):  30.0 %     |                       |                       |                       |
| --------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| timestep  1 : 0.49509 | timestep  2 : 0.48218 | timestep  3 : 0.46412 | timestep  4 : 0.56761 | timestep  5 : 0.55554 |
| timestep  6 : 0.54993 | timestep  7 : 0.51521 | timestep  8 : 0.40278 | timestep  9 : 0.50833 | timestep  10 : 0.625  |
| timestep  11 : 0.75   | timestep  12 : 0.75   | timestep  13 : 0.5    | timestep  14 : 0.25   | timestep  15 : 0.0    |
| timestep  16 : 0.0    | timestep  17 : 0.0    | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0    |
| timestep  21 : 0.0    | timestep  22 : 0.0    | timestep  23 : 0.0    |                       |                       |

| Input: WO_Activity     | Input(%):  40.0 %     |                       |                       |                       |
| ---------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| timestep  1 : 0.67872  | timestep  2 : 0.60549 | timestep  3 : 0.61025 | timestep  4 : 0.65273 | timestep  5 : 0.47169 |
| timestep  6 : 0.54984  | timestep  7 : 0.53804 | timestep  8 : 0.525   | timestep  9 : 0.5     | timestep  10 : 0.625  |
| timestep  11 : 0.91667 | timestep  12 : 0.5    | timestep  13 : 0.0    | timestep  14 : 0.0    | timestep  15 : 0.0    |
| timestep  16 : 0.0     | timestep  17 : 0.0    | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0    |
| timestep  21 : 0.0     | timestep  22 : 0.0    | timestep  23 : 0.0    |                       |                       |

| Input: WO_Activity    | Input(%):  50.0 %     |                       |                       |                       |
| --------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| timestep  1 : 0.77887 | timestep  2 : 0.71031 | timestep  3 : 0.71102 | timestep  4 : 0.72073 | timestep  5 : 0.47238 |
| timestep  6 : 0.67063 | timestep  7 : 0.66389 | timestep  8 : 0.6     | timestep  9 : 1.0     | timestep  10 : 0.75   |
| timestep  11 : 0.5    | timestep  12 : 0.25   | timestep  13 : 0.0    | timestep  14 : 0.0    | timestep  15 : 0.0    |
| timestep  16 : 0.0    | timestep  17 : 0.0    | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0    |
| timestep  21 : 0.0    | timestep  22 : 0.0    | timestep  23 : 0.0    |                       |                       |

### With activity

***

| Input: W_Activity     | Input(%):  10.0 %     |                       |                       |                        |
| --------------------- | --------------------- | --------------------- | --------------------- | ---------------------- |
| timestep  1 : 0.46406 | timestep  2 : 0.49786 | timestep  3 : 0.4502  | timestep  4 : 0.53528 | timestep  5 : 0.46148  |
| timestep  6 : 0.65156 | timestep  7 : 0.60996 | timestep  8 : 0.57623 | timestep  9 : 0.39673 | timestep  10 : 0.75357 |
| timestep  11 : 0.275  | timestep  12 : 0.6875 | timestep  13 : 0.4    | timestep  14 : 0.25   | timestep  15 : 0.25    |
| timestep  16 : 0.0    | timestep  17 : 0.0    | timestep  18 : 0.0    | timestep  19 : 0.25   | timestep  20 : 0.0     |
| timestep  21 : 0.0    | timestep  22 : 0.25   | timestep  23 : 0.0    |                       |                        |

| Input: W_Activity      | Input(%):  20.0 %     |                       |                       |                       |
| ---------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| timestep  1 : 0.52702  | timestep  2 : 0.55295 | timestep  3 : 0.55166 | timestep  4 : 0.58225 | timestep  5 : 0.71452 |
| timestep  6 : 0.63848  | timestep  7 : 0.50439 | timestep  8 : 0.74632 | timestep  9 : 0.35385 | timestep  10 : 0.775  |
| timestep  11 : 0.29167 | timestep  12 : 0.5    | timestep  13 : 0.25   | timestep  14 : 0.5    | timestep  15 : 0.0    |
| timestep  16 : 0.0     | timestep  17 : 0.25   | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0    |
| timestep  21 : 0.0     | timestep  22 : 0.0    | timestep  23 : 0.0    |                       |                       |

| Input: W_Activity     | Input(%):  30.0 %     |                       |                       |                       |
| --------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| timestep  1 : 0.60637 | timestep  2 : 0.53526 | timestep  3 : 0.53442 | timestep  4 : 0.64833 | timestep  5 : 0.64619 |
| timestep  6 : 0.56243 | timestep  7 : 0.74034 | timestep  8 : 0.51786 | timestep  9 : 0.65278 | timestep  10 : 0.225  |
| timestep  11 : 0.625  | timestep  12 : 0.75   | timestep  13 : 0.0    | timestep  14 : 0.0    | timestep  15 : 0.0    |
| timestep  16 : 0.0    | timestep  17 : 0.0    | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0    |
| timestep  21 : 0.0    | timestep  22 : 0.0    | timestep  23 : 0.0    |                       |                       |

| Input: W_Activity      | Input(%):  40.0 %     |                       |                       |                        |
| ---------------------- | --------------------- | --------------------- | --------------------- | ---------------------- |
| timestep  1 : 0.70637  | timestep  2 : 0.64513 | timestep  3 : 0.67322 | timestep  4 : 0.63518 | timestep  5 : 0.56661  |
| timestep  6 : 0.51789  | timestep  7 : 0.56787 | timestep  8 : 0.36111 | timestep  9 : 0.5375  | timestep  10 : 0.76667 |
| timestep  11 : 0.66667 | timestep  12 : 0.5    | timestep  13 : 0.0    | timestep  14 : 0.0    | timestep  15 : 0.0     |
| timestep  16 : 0.0     | timestep  17 : 0.0    | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0     |
| timestep  21 : 0.0     | timestep  22 : 0.0    | timestep  23 : 0.0    |                       |                        |

| Input: W_Activity     | Input(%):  50.0 %     |                       |                      |                       |
| --------------------- | --------------------- | --------------------- | -------------------- | --------------------- |
| timestep  1 : 0.78758 | timestep  2 : 0.70342 | timestep  3 : 0.78805 | timestep  4 : 0.8112 | timestep  5 : 0.54728 |
| timestep  6 : 0.72123 | timestep  7 : 0.58889 | timestep  8 : 0.5625  | timestep  9 : 0.6875 | timestep  10 : 0.5    |
| timestep  11 : 0.5    | timestep  12 : 0.5    | timestep  13 : 0.0    | timestep  14 : 0.0   | timestep  15 : 0.0    |
| timestep  16 : 0.0    | timestep  17 : 0.0    | timestep  18 : 0.0    | timestep  19 : 0.0   | timestep  20 : 0.0    |
| timestep  21 : 0.0    | timestep  22 : 0.0    | timestep  23 : 0.0    |                      |                       |

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

| Input: W_Activity     | Input(%):  10.0 %     |                       |                       |                        |
| --------------------- | --------------------- | --------------------- | --------------------- | ---------------------- |
| timestep  1 : 0.46406 | timestep  2 : 0.49786 | timestep  3 : 0.4502  | timestep  4 : 0.53528 | timestep  5 : 0.46148  |
| timestep  6 : 0.65156 | timestep  7 : 0.60996 | timestep  8 : 0.57623 | timestep  9 : 0.39673 | timestep  10 : 0.75357 |
| timestep  11 : 0.275  | timestep  12 : 0.6875 | timestep  13 : 0.4    | timestep  14 : 0.25   | timestep  15 : 0.25    |
| timestep  16 : 0.0    | timestep  17 : 0.0    | timestep  18 : 0.0    | timestep  19 : 0.25   | timestep  20 : 0.0     |
| timestep  21 : 0.0    | timestep  22 : 0.25   | timestep  23 : 0.0    |                       |                        |

| Input: W_Activity      | Input(%):  20.0 %     |                       |                       |                       |
| ---------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| timestep  1 : 0.52702  | timestep  2 : 0.55295 | timestep  3 : 0.55166 | timestep  4 : 0.58225 | timestep  5 : 0.71452 |
| timestep  6 : 0.63848  | timestep  7 : 0.50439 | timestep  8 : 0.74632 | timestep  9 : 0.35385 | timestep  10 : 0.775  |
| timestep  11 : 0.29167 | timestep  12 : 0.5    | timestep  13 : 0.25   | timestep  14 : 0.5    | timestep  15 : 0.0    |
| timestep  16 : 0.0     | timestep  17 : 0.25   | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0    |
| timestep  21 : 0.0     | timestep  22 : 0.0    | timestep  23 : 0.0    |                       |                       |

| Input: W_Activity     | Input(%):  30.0 %     |                       |                       |                       |
| --------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| timestep  1 : 0.60637 | timestep  2 : 0.53526 | timestep  3 : 0.53442 | timestep  4 : 0.64833 | timestep  5 : 0.64619 |
| timestep  6 : 0.56243 | timestep  7 : 0.74034 | timestep  8 : 0.51786 | timestep  9 : 0.65278 | timestep  10 : 0.225  |
| timestep  11 : 0.625  | timestep  12 : 0.75   | timestep  13 : 0.0    | timestep  14 : 0.0    | timestep  15 : 0.0    |
| timestep  16 : 0.0    | timestep  17 : 0.0    | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0    |
| timestep  21 : 0.0    | timestep  22 : 0.0    | timestep  23 : 0.0    |                       |                       |

| Input: W_Activity      | Input(%):  40.0 %     |                       |                       |                        |
| ---------------------- | --------------------- | --------------------- | --------------------- | ---------------------- |
| timestep  1 : 0.70637  | timestep  2 : 0.64513 | timestep  3 : 0.67322 | timestep  4 : 0.63518 | timestep  5 : 0.56661  |
| timestep  6 : 0.51789  | timestep  7 : 0.56787 | timestep  8 : 0.36111 | timestep  9 : 0.5375  | timestep  10 : 0.76667 |
| timestep  11 : 0.66667 | timestep  12 : 0.5    | timestep  13 : 0.0    | timestep  14 : 0.0    | timestep  15 : 0.0     |
| timestep  16 : 0.0     | timestep  17 : 0.0    | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0     |
| timestep  21 : 0.0     | timestep  22 : 0.0    | timestep  23 : 0.0    |                       |                        |

| Input: W_Activity     | Input(%):  50.0 %     |                       |                      |                       |
| --------------------- | --------------------- | --------------------- | -------------------- | --------------------- |
| timestep  1 : 0.78758 | timestep  2 : 0.70342 | timestep  3 : 0.78805 | timestep  4 : 0.8112 | timestep  5 : 0.54728 |
| timestep  6 : 0.72123 | timestep  7 : 0.58889 | timestep  8 : 0.5625  | timestep  9 : 0.6875 | timestep  10 : 0.5    |
| timestep  11 : 0.5    | timestep  12 : 0.5    | timestep  13 : 0.0    | timestep  14 : 0.0   | timestep  15 : 0.0    |
| timestep  16 : 0.0    | timestep  17 : 0.0    | timestep  18 : 0.0    | timestep  19 : 0.0   | timestep  20 : 0.0    |
| timestep  21 : 0.0    | timestep  22 : 0.0    | timestep  23 : 0.0    |                      |                       |





## TCN

***

## 1. loss='categorical_crossentropy', optimizer='adam', activation = "softmax"

| Input: WO_Activity    | Input(%):  10.0 %      |                       |                       |                       |
| --------------------- | ---------------------- | --------------------- | --------------------- | --------------------- |
| timestep  1 : 0.20103 | timestep  2 : 0.29377  | timestep  3 : 0.24269 | timestep  4 : 0.36751 | timestep  5 : 0.35118 |
| timestep  6 : 0.56988 | timestep  7 : 0.45829  | timestep  8 : 0.43904 | timestep  9 : 0.36905 | timestep  10 : 0.4496 |
| timestep  11 : 0.5125 | timestep  12 : 0.59375 | timestep  13 : 0.5    | timestep  14 : 0.25   | timestep  15 : 0.0    |
| timestep  16 : 0.25   | timestep  17 : 0.0     | timestep  18 : 0.0    | timestep  19 : 0.25   | timestep  20 : 0.25   |
| timestep  21 : 0.0    | timestep  22 : 0.25    | timestep  23 : 0.0    |                       |                       |

| Input: WO_Activity     | Input(%):  20.0 %     |                        |                       |                        |
| ---------------------- | --------------------- | ---------------------- | --------------------- | ---------------------- |
| timestep  1 : 0.34494  | timestep  2 : 0.42376 | timestep  3 : 0.37537  | timestep  4 : 0.42242 | timestep  5 : 0.53135  |
| timestep  6 : 0.56858  | timestep  7 : 0.30839 | timestep  8 : 0.45003  | timestep  9 : 0.43419 | timestep  10 : 0.62917 |
| timestep  11 : 0.47917 | timestep  12 : 0.5    | timestep  13 : 0.66667 | timestep  14 : 0.5    | timestep  15 : 0.25    |
| timestep  16 : 0.25    | timestep  17 : 0.25   | timestep  18 : 0.0     | timestep  19 : 0.0    | timestep  20 : 0.0     |
| timestep  21 : 0.0     | timestep  22 : 0.0    | timestep  23 : 0.0     |                       |                        |

| Input: WO_Activity     | Input(%):  30.0 %      |                       |                       |                       |
| ---------------------- | ---------------------- | --------------------- | --------------------- | --------------------- |
| timestep  1 : 0.50479  | timestep  2 : 0.47726  | timestep  3 : 0.48855 | timestep  4 : 0.55682 | timestep  5 : 0.52397 |
| timestep  6 : 0.57199  | timestep  7 : 0.51521  | timestep  8 : 0.40278 | timestep  9 : 0.425   | timestep  10 : 0.625  |
| timestep  11 : 0.91667 | timestep  12 : 0.91667 | timestep  13 : 0.5    | timestep  14 : 0.25   | timestep  15 : 0.0    |
| timestep  16 : 0.0     | timestep  17 : 0.0     | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0    |
| timestep  21 : 0.0     | timestep  22 : 0.0     | timestep  23 : 0.0    |                       |                       |

| Input: WO_Activity     | Input(%):  40.0 %     |                       |                       |                        |
| ---------------------- | --------------------- | --------------------- | --------------------- | ---------------------- |
| timestep  1 : 0.63935  | timestep  2 : 0.61267 | timestep  3 : 0.62892 | timestep  4 : 0.70464 | timestep  5 : 0.50294  |
| timestep  6 : 0.53105  | timestep  7 : 0.51531 | timestep  8 : 0.375   | timestep  9 : 0.4375  | timestep  10 : 0.79167 |
| timestep  11 : 0.91667 | timestep  12 : 0.5    | timestep  13 : 0.0    | timestep  14 : 0.0    | timestep  15 : 0.0     |
| timestep  16 : 0.0     | timestep  17 : 0.0    | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0     |
| timestep  21 : 0.0     | timestep  22 : 0.0    | timestep  23 : 0.0    |                       |                        |

| Input: WO_Activity    | Input(%):  50.0 %     |                       |                       |                       |
| --------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| timestep  1 : 0.7392  | timestep  2 : 0.71368 | timestep  3 : 0.77832 | timestep  4 : 0.72999 | timestep  5 : 0.60712 |
| timestep  6 : 0.74405 | timestep  7 : 0.63889 | timestep  8 : 0.6625  | timestep  9 : 0.9375  | timestep  10 : 0.75   |
| timestep  11 : 0.25   | timestep  12 : 0.5    | timestep  13 : 0.0    | timestep  14 : 0.0    | timestep  15 : 0.0    |
| timestep  16 : 0.0    | timestep  17 : 0.0    | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0    |
| timestep  21 : 0.0    | timestep  22 : 0.0    | timestep  23 : 0.0    |                       |                       |

|      |      |      |      |      |
| ---- | ---- | ---- | ---- | ---- |
|      |      |      |      |      |
|      |      |      |      |      |
|      |      |      |      |      |
|      |      |      |      |      |
|      |      |      |      |      |

Transformer without

| Input: WO_Activity    | Input(%):  10.0 %      |                        |                       |                        |
| --------------------- | ---------------------- | ---------------------- | --------------------- | ---------------------- |
| timestep  1 : 0.29503 | timestep  2 : 0.3333   | timestep  3 : 0.25092  | timestep  4 : 0.3629  | timestep  5 : 0.29109  |
| timestep  6 : 0.5697  | timestep  7 : 0.45984  | timestep  8 : 0.61259  | timestep  9 : 0.36101 | timestep  10 : 0.61071 |
| timestep  11 : 0.425  | timestep  12 : 0.34375 | timestep  13 : 0.51667 | timestep  14 : 0.25   | timestep  15 : 0.0     |
| timestep  16 : 0.25   | timestep  17 : 0.0     | timestep  18 : 0.0     | timestep  19 : 0.25   | timestep  20 : 0.0     |
| timestep  21 : 0.0    | timestep  22 : 0.25    | timestep  23 : 0.0     |                       |                        |

| Input: WO_Activity     | Input(%):  20.0 %    |                        |                       |                       |
| ---------------------- | -------------------- | ---------------------- | --------------------- | --------------------- |
| timestep  1 : 0.49508  | timestep  2 : 0.4101 | timestep  3 : 0.36303  | timestep  4 : 0.48603 | timestep  5 : 0.51916 |
| timestep  6 : 0.58204  | timestep  7 : 0.3596 | timestep  8 : 0.585    | timestep  9 : 0.38419 | timestep  10 : 0.775  |
| timestep  11 : 0.60417 | timestep  12 : 0.75  | timestep  13 : 0.33333 | timestep  14 : 0.75   | timestep  15 : 0.25   |
| timestep  16 : 0.0     | timestep  17 : 0.25  | timestep  18 : 0.0     | timestep  19 : 0.0    | timestep  20 : 0.0    |
| timestep  21 : 0.0     | timestep  22 : 0.0   | timestep  23 : 0.0     |                       |                       |

| Input: WO_Activity    | Input(%):  30.0 %      |                       |                       |                       |
| --------------------- | ---------------------- | --------------------- | --------------------- | --------------------- |
| timestep  1 : 0.61318 | timestep  2 : 0.45522  | timestep  3 : 0.44341 | timestep  4 : 0.49165 | timestep  5 : 0.52324 |
| timestep  6 : 0.47487 | timestep  7 : 0.40186  | timestep  8 : 0.46032 | timestep  9 : 0.49444 | timestep  10 : 0.4125 |
| timestep  11 : 0.625  | timestep  12 : 0.66667 | timestep  13 : 0.5    | timestep  14 : 0.25   | timestep  15 : 0.0    |
| timestep  16 : 0.0    | timestep  17 : 0.0     | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0    |
| timestep  21 : 0.0    | timestep  22 : 0.0     | timestep  23 : 0.0    |                       |                       |

| Input: WO_Activity    | Input(%):  40.0 %     |                       |                       |                        |
| --------------------- | --------------------- | --------------------- | --------------------- | ---------------------- |
| timestep  1 : 0.70811 | timestep  2 : 0.57215 | timestep  3 : 0.5     | timestep  4 : 0.52336 | timestep  5 : 0.61553  |
| timestep  6 : 0.60623 | timestep  7 : 0.51389 | timestep  8 : 0.42183 | timestep  9 : 0.5     | timestep  10 : 0.79167 |
| timestep  11 : 0.25   | timestep  12 : 0.0    | timestep  13 : 0.0    | timestep  14 : 0.0    | timestep  15 : 0.0     |
| timestep  16 : 0.0    | timestep  17 : 0.0    | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0     |
| timestep  21 : 0.0    | timestep  22 : 0.0    | timestep  23 : 0.0    |                       |                        |

| Input: WO_Activity    | Input(%):  50.0 %     |                       |                       |                       |
| --------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| timestep  1 : 0.76927 | timestep  2 : 0.63162 | timestep  3 : 0.61333 | timestep  4 : 0.62192 | timestep  5 : 0.36102 |
| timestep  6 : 0.57837 | timestep  7 : 0.675   | timestep  8 : 0.51111 | timestep  9 : 0.6875  | timestep  10 : 0.5    |
| timestep  11 : 0.25   | timestep  12 : 0.0    | timestep  13 : 0.0    | timestep  14 : 0.0    | timestep  15 : 0.0    |
| timestep  16 : 0.0    | timestep  17 : 0.0    | timestep  18 : 0.0    | timestep  19 : 0.0    | timestep  20 : 0.0    |
| timestep  21 : 0.0    | timestep  22 : 0.0    | timestep  23 : 0.0    |                       |                       |











Model 3: Lstm
Model:  LSTM      With_Activity:  No Goal accuracy
0.83897  0.92506  0.95369  0.98682  1.0  
Model:  LSTM      With_Activity:  No Next action segment accuracy
0.68983  0.75618  0.78328  0.78767  0.85287  

Model:  LSTM      With_Activity:  No Goal accuracy
0.84746  0.92969  0.95369  0.99007  1.0  1.0  1.0  1.0  1.0  
Model:  LSTM      With_Activity:  No Next action segment accuracy
0.68951  0.74217  0.77708  0.78639  0.85469  0.85506  0.86214  0.87718  0.91791  



accuracy :  443 / 503 = 0.8807157057654076
Model:  TCN      With_Activity:  No Goal accuracy
0.84785  0.928  0.95503  0.9876  0.9946  
Model:  TCN      With_Activity:  No Next action segment accuracy
0.73295  0.75415  0.76942  0.78953  0.84732  

Model:  TCN      With_Activity:  No Goal accuracy
0.8442  0.927  0.9503  0.98542  1.0  1.0  1.0  1.0  1.0  
Model:  TCN      With_Activity:  No Next action segment accuracy
0.71785  0.76558  0.78705  0.79674  0.84694  0.85714  0.85889  0.88595  0.90539 

Model:  Transformer      With_Activity:  No Goal accuracy
0.84517  0.93235  0.95946  0.99007  1.0  
Model:  Transformer      With_Activity:  No Next action segment accuracy
0.72026  0.75578  0.76748  0.78532  0.85234 

Model:  Transformer      With_Activity:  No Goal accuracy
0.84525  0.92755  0.96065  0.99007  1.0  1.0  1.0  1.0  1.0  
Model:  Transformer      With_Activity:  No Next action segment accuracy
0.72453  0.75736  0.7736  0.79594  0.84205  0.85538  0.86157  0.8721  0.90466  





Model 1:

 Model:  LSTM      With_Activity:  No  Goal accuracy
0.56585  0.73605  0.78259  0.80516  0.85374  
Model:  LSTM      With_Activity:  No Next action segment accuracy
0.77651  0.92723  0.95768  0.9876  0.99769  



Model:  TCN      With_Activity:  No         action accuracy
0.73515  0.72962  0.76201  0.78995  0.84733  
Model:  TCN      With_Activity:  No        goal
0.83203  0.93103  0.95593  0.99007  0.99884  

Model:  Transformer      With_Activity:  No         action accuracy
0.72744  0.73834  0.76137  0.78949  0.84894  
Model:  Transformer      With_Activity:  No        goal
0.85564  0.92617  0.96256  0.98682  1.0 





model3: last
Model:  LSTM      With_Activity:  No         action accuracy
0.66615  0.72019  0.77246  0.78519  0.85001  
Model:  LSTM      With_Activity:  No        goal
0.84439  0.93328  0.96166  0.97732  0.99872  

Model:  TCN      With_Activity:  No         action accuracy
0.69952  0.74102  0.77506  0.79056  0.85438  
Model:  TCN      With_Activity:  No        goal
0.85546  0.92397  0.94735  0.99007  1.0  

accuracy :  443 / 503 = 0.8807157057654076
accuracy :  503 / 503 = 1.0
Model:  LSTM      With_Activity:  No        goal
0.69945  0.73893  0.76992  0.79352  0.85041  
Model:  LSTM      With_Activity:  No         action accuracy
0.85355  0.92466  0.95961  0.99007  1.0  

Process finished with exit code 0