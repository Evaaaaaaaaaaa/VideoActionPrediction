import matplotlib.ticker as ticker
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
# params = {"ytick.color" : "w",
#           "xtick.color" : "w",
#           "axes.labelcolor" : "w",
#           "axes.edgecolor" : "w",
#           "legend.facecolor" : "w"}
# plt.rcParams.update(params)
#model1 0.1 - 0.5
lstm_wo1 = [0.6974, 0.73245, 0.77796, 0.80716, 0.85775]
lstm_w1 = [0.76193, 0.78089,0.79197, 0.79339, 0.85446]
tcn_wo1 = [0.73097, 0.73196, 0.77432, 0.78562, 0.84773]
tcn_w1 = [0.78831, 0.77984,0.79246, 0.80601, 0.86044]
tran_wo1 = [0.71876, 0.73631, 0.76636, 0.78604, 0.844]
tran_w1 = [0.8086, 0.78829,0.77907, 0.8036, 0.8562]


# model2 0.1-0.5
lstm_action2 = [0.69048, 0.73881 ,0.77042 , 0.79194,  0.84644]
lstm_goal2= [0.85221,  0.92272 , 0.95782,  0.99007 , 1.0]
tcn_action2 = [0.72003 , 0.7427 , 0.76454,  0.79879 , 0.82614]
tcn_goal2 = [0.83257,  0.92654,  0.95562,  0.99007,  0.99074]
tran_action2 = [0.72522 , 0.73052,  0.76374,  0.7914,  0.84009]
tran_goal2 = [0.83839,  0.92135 , 0.96126 , 0.99007 , 1.0 ]



# model3:
lstm_action3 = [0.69845, 0.72069,0.75895 , 0.80078, 0.84725]
lstm_goal3= [0.84439, 0.93328,0.96166,0.97732,0.99872]
tcn_action3 = [0.69952 ,0.74102, 0.77506,  0.79056 , 0.85438]
tcn_goal3 = [0.85546,0.92397,0.94735,0.99007,1]
tran_action3 = [0.71253 , 0.7394,  0.75874, 0.79422, 0.84875]
tran_goal3 = [0.8524,0.91933,0.95843,0.99007,1]

# model4:
lstm_action4 = [0.69845,0.72069,0.75895,0.80078,0.84725]
lstm_goal4= [0.84439,0.93328, 0.96166,0.97732,0.99872]
tcn_action4 = [0.69952,0.74102,0.77506,0.79056,0.85438]
tcn_goal4 = [0.85546,0.92397,0.94735,0.99007,1]
tran_action4 = [0.71253,0.7394,0.75874,0.79422,0.84875]
tran_goal4 = [0.8524,0.91933,0.95843,0.99007,1]

# model5:
lstm_action5 = [0.71545,0.74808,0.76726,0.80136,0.86339]
lstm_goal5= [0.85301,0.92338, 0.95754,0.99007,1]
tcn_action5 = [0.71545,0.74808,0.76726,0.80136,0.863398]
tcn_goal5 = [0.85134,0.91569,0.95562,0.98542,1]
tran_action5 = [0.71918 , 0.74931,  0.76016 , 0.78685,  0.86462  ]
tran_goal5 = [0.85103,0.92382,0.95857,0.99007,1]



# x = [0.1, 0.2, 0.3, 0.4, 0.5]
# plt.plot(np.arange(0.1, 0.6, 0.1), lstm_wo1, '-ok', color='g', label='baseline')
# plt.plot(np.arange(0.1, 0.6, 0.1), lstm_action2, '-ok', color='r', label='with activity supervision')
plt.plot(np.arange(0.1, 0.6, 0.1), lstm_goal2, '-ok', color='blue', label='predicted activity')
# TCN Train ground-truth label
# separated network
# joint network
# with activity supervision
x_ticks = np.arange(0.1, 0.6, 0.1)
plt.xticks(x_ticks)
plt.title('The accuracy of the predicted goal for method 2 using LSTM')
plt.xlabel('Input Proportion')
plt.ylabel('Acc')

legend = plt.legend(loc= "upper left")
# legend.get_frame().set_facecolor((0, 0, 0, 0))


plt.savefig('lstm_goal.png', transparent=True)
plt.show()


