# todo: ignore video if there is no next action
# todo: model3, output both goal and next action
# import
from sklearn import metrics
import scipy.io
import os
import numpy as np
import copy
import statistics
from operator import add


# import data
def import_data():
    data_dict = {}
    path = 'groundTruthmat/'
    files = os.listdir(path)
    for file in files:
        read = scipy.io.loadmat(path + file)
        data_dict[file] = read['labseqid']

    return data_dict


# cut frames with SIL action
def cut_zeros(data_dict):
    for filename, frames in data_dict.items():
        while frames[0] == 0:
            frames = frames[1:]
        while frames[-1] == 0:
            frames = frames[:-1]
        data_dict[filename] = frames
    return data_dict


# find max action length
def find_max_action_length(data_dict):
    # to minimize input size, convert frames to actions first
    temp_dict = copy.deepcopy(data_dict)
    temp_dict = frames_to_action(temp_dict)
    max_action_len = 0
    for filename, action in temp_dict.items():
        if len(action) > max_action_len:
            max_action_len = len(action)
    return max_action_len


# def get_input_x_y(data_dict, trainPortion):
#     new_dict = {}
#     train_y = []
#     for filename, frames in data_dict.items():
#         frameLen = len(frames)
#
#         inputLen = round(frameLen * trainPortion)
#         inputFrames = frames[:inputLen]
#         for frame in frames[inputLen:]:
#             #             print(frame, y)
#             # if there is next action, add to train x and train y.
#             if frame[0] != frames[inputLen - 1][0]:
#                 train_y.append([frame[0]])
#                 new_dict[filename] = inputFrames
#                 break
#
#     return new_dict, train_y


# get trainY (action class) for trainX
def get_input_y(data_dict, input_dict):
    train_y = []
    for filename, frames in data_dict.items():
        frame_len = len(input_dict[filename])
        y = [input_dict[filename][frame_len - 1][0]]
        for frame in frames[frame_len:]:
            #             print(frame, y)
            if frame[0] != y[0]:
                y[0] = frame[0]
                break
        train_y.append(y)
    return train_y


# get input x (action segeent), y (next action segement) and goal
def get_input_x_y(data_dict, trainPortion, is_goal):
    # goal: 1 --> next action segment
    # goal: 2 --> activity
    new_dict = {}
    train_goal_y = []
    train_action_y = []
    if is_goal:
        for filename, frames in data_dict.items():
            has_next = 0
            frameLen = len(frames)
            inputLen = round(frameLen * trainPortion)
            inputFrames = frames[:inputLen]
            next = [frames[inputLen - 1][0]]
            for frame in frames[inputLen:]:
                #             print(frame, y)
                if frame[0] != next[0]:
                    has_next = 1
                    next[0] = frame[0]

                    break
            if has_next:
                activity = [get_activity(filename)]
                train_goal_y.append(activity)
                train_action_y.append(next)
                new_dict[filename] = inputFrames
    return new_dict, train_goal_y, train_action_y


# get input x based on given proportion
def get_input_x(data_dict, proportion):
    new_dict = {}
    for filename, frames in data_dict.items():
        frameLen = len(frames)
        inputLen = round(frameLen * proportion)
        inputFrames = frames[:inputLen]
        new_dict[filename] = inputFrames
    return new_dict


def add_activity(input_dict):
    activities = {"cereals": 1, "coffee": 2, "friedegg": 3, "milk": 4, "salat": 5, "sandwich": 6, "tea": 7,
                  "pancake": 8, "scrambledegg": 9, "juice": 10}
    one_hot = {}
    temp_dict = copy.deepcopy(input_dict)
    for i in range(len(activities)):
        encode = len(activities) * [0]
        encode[i] = 1
        one_hot[i + 1] = encode

    for filename, actions in temp_dict.items():
        activity = get_activity(filename)
        for i in range(len(actions)):
            temp_dict[filename][i] = temp_dict[filename][i] + one_hot[activities[activity]]

    return temp_dict


# get activity as "string"
def get_activity(filename):
    activities = {"cereals": 1, "coffee": 2, "friedegg": 3, "milk": 4, "salat": 5, "sandwich": 6, "tea": 7,
                  "pancake": 8, "scrambledegg": 9, "juice": 10}
    # convert filename to int
    activity = activities[(filename.split("_", 3)[-1]).split(".")[0]]
    return activity


def frames_to_action(input_dict):
    for filename, frames in input_dict.items():
        action_list = []
        for frame in frames:
            if len(action_list) == 0:
                action_list.append(frame[0])
            else:
                if frame[0] != action_list[-1]:
                    action_list.append(frame[0])
        input_dict[filename] = action_list
    return input_dict


# add padding, each video has same length of action input
def add_padding_to_x(data_dict, maxLen):
    for filename, frames in data_dict.items():
        data_dict[filename] = (maxLen - len(frames)) * [-1] + frames
    return data_dict


def get_action_label():
    # create frame_label for one hot encoding
    action_label = {}
    for i in range(1, 48):
        action_label[i] = 47 * [0]
        action_label[i][i - 1] = 1
    action_label[-1] = 47 * [0]
    return action_label


# perform one-hot encoding to x
def feature_encoding(input_dict, action_label):
    for filename, frames in input_dict.items():
        # get corresponding one-hot encode
        new = []
        for each in frames:
            new.append(action_label[each])
        input_dict[filename] = new
    return input_dict


def label_encoding(train_y, action_label, is_goal):
    new_trainY = []
    if is_goal:
        one_hot = {}
        temp_y = copy.deepcopy(train_y)
        for i in range(10):
            encode = 10 * [0]
            encode[i] = 1
            one_hot[i + 1] = encode

        for i in range(len(temp_y)):
            temp_y[i] = one_hot[temp_y[i][0]]
        return temp_y

    else:
        for frames in train_y:
            new_trainY.append(action_label[frames[0]])
    return new_trainY


def get_sample_weight(train_x):
    sample_weight = []
    for actions in train_x:
        sample_weight.append([])
        for each in actions:
            if each == len(each) * [0]:
                sample_weight[-1].append(0)
            else:
                sample_weight[-1].append(1)
    return sample_weight


from keras.models import Sequential
from keras.layers import LSTM
from tcn import TCN
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input, Model

#
def lstm_model(max_action_len):
    i = Input(shape=(max_action_len, 47))
    # o = LSTM(512, return_sequences=True)(i)
    # o = LSTM(256, return_sequences=True)(o)
    o = LSTM(128)(i)
    o1 = Dense(10, activation="softmax")(o)
    o1_repeat = RepeatVector(max_action_len)(o1)
    # # todo: i + o1_repeat concatenate
    o1_repeat = tf.concat([i, o1_repeat], -1)
    # o2 = LSTM(512, return_sequences=True)(o1_repeat)
    # o2 = LSTM(256, return_sequences=True)(o1_repeat)
    o2 = LSTM(128)(o1_repeat)
    o2 = Dense(47, activation="softmax")(o2)
    # o2 = Dense(47, activation="softmax")(o1)
    model = Model(inputs=[i], outputs=[o2, o1])
    model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'], optimizer='adam', metrics=['accuracy'],
                  sample_weight_mode="temporal")
    model.summary()
    return model


def tcn_model(max_action_len):
    i = Input(shape=(max_action_len, 47))
    o = TCN(128, return_sequences=True)(i)  # The TCN layers are here.
    o = TCN(64, return_sequences=False)(o)
    o1 = Dense(10, activation="softmax")(o)
    o1_repeat = RepeatVector(max_action_len)(o1)
    # # todo: i + o1_repeat concatenate
    o1_repeat = tf.concat([i, o1_repeat], -1)
    o2 = TCN(64, return_sequences=False)(o1_repeat)
    o2 = Dense(47, activation="softmax")(o2)
    model = Model(inputs=[i], outputs=[o2, o1])
    model.compile(optimizer='adam', loss=['categorical_crossentropy', 'categorical_crossentropy'], metrics=['accuracy'])
    model.summary()
    return model


# transformer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, RepeatVector
from keras.layers import Dropout, Masking, TimeDistributed, Activation
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.projection_dim)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )
        output = self.combine_heads(concat_attention)
        return output


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, 1)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class PositionEmbedding(layers.Layer):
    def __init__(self, maxlen, embed_dim):
        super(PositionEmbedding, self).__init__()
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)

        return positions


def transformer_model(max_action_len):
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer
    inputs = layers.Input(shape=(max_action_len, 47))
    transformer_block1 = TransformerBlock(47, 1, ff_dim)
    i = transformer_block1(inputs)
    # x = layers.GlobalAveragePooling1D()(x)
    # x = layers.Dropout(0.1)(x)
    i = layers.Dense(128, activation="relu")(i)
    # # x = layers.Dense(64, activation="relu")(x)
    i = layers.Flatten()(i)
    o1 = layers.Dense(10, activation="softmax")(i)
    o1_repeat = RepeatVector(max_action_len)(o1)
    o1_repeat = tf.concat([inputs, o1_repeat], -1)
    transformer_block2 = TransformerBlock(57, 1, ff_dim)
    o2 = transformer_block2(o1_repeat)
    o2 = layers.Dense(128, activation="relu")(o2)
    o2 = layers.Flatten()(o2)
    o2 = layers.Dense(47, activation="softmax")(o2)
    model = keras.Model(inputs=inputs, outputs=[o2, o1])
    model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'], optimizer='adam', metrics=['accuracy'],
                  sample_weight_mode="temporal")
    # sample_weight_mode = "temporal"
    model.summary()
    return model


# cross validation
def train_model(trainX, trainY, model, sample_weight):
    print(np.array(trainX).shape, np.array(trainY[0]).shape,np.array(trainY[1]).shape )
    model.fit(np.array(trainX), [np.array(trainY[0]), np.array(trainY[1])], epochs=20, batch_size=128)


# def evaluation(testX, testY, model, max_timesteps):
#     predictions = model.predict(testX)
#     results = []
#     count = 0
#     for i in range(len(predictions)):
#         result = np.array([0] * 47)
#         index = predictions[i].argmax(axis=-1)
#         result[index] = 1
#         pre = metrics.accuracy_score(result, testY[i])
#         if pre == 1:
#             count += 1
#     print(count / len(testY))
#     return count / len(testY)

def evaluation(testX, testY, model, max_timesteps):
    predictions = model.predict(testX)
    predicted_action = predictions[0]
    predicted_goal = predictions[1]

    action_results = []
    action_accuracy_list = []
    for i in range(len(predicted_action)):
        each_video = []
        action_result = np.array([0] * 47)
        index = predicted_action[i].argmax(axis=-1)
        action_result[index] = 1
        action_results.append(action_result)
    correct = 0
    # loop each video
    for j in range(len(testY[0])):
        if (testY[0][j] == action_results[j]).all():
            correct += 1

    action_acc= correct / len(action_results)
    print("accuracy", ": ", correct, "/", len(action_results), "=", correct / len(action_results))

    goal_results = []
    goal_accuracy_list = []
    for i in range(len(predicted_action)):
        each_video = []
        goal_result = np.array([0] * 10)
        index = predicted_goal[i].argmax(axis=-1)
        goal_result[index] = 1
        goal_results.append(goal_result)
    correct = 0
    # loop each video
    for j in range(len(testY[1])):
        if (testY[1][j] == goal_results[j]).all():
            correct += 1
    goal_acc= correct / len(goal_results)
    print("accuracy", ": ", correct, "/", len(action_results), "=", correct / len(action_results))
    return action_acc, goal_acc


def cross_validation(input_dict, goal_y, action_y, model, max_timesteps, is_goal):
    file_count = 0
    s1_x, s2_x, s3_x, s4_x = [], [], [], []
    s1_goal_y, s2_goal_y, s3_goal_y, s4_goal_y = [], [], [], []
    s1_action_y, s2_action_y, s3_action_y, s4_action_y = [], [], [], []
    # s1: P03 – P15
    # s2: P16 – P28
    # s3: P29 – P41
    # s4: P42 – P54
    count = 0
    print(len(input_dict))

    for filename, frames in input_dict.items():
        if int(filename[1:3]) <= 15:
            s1_x.append(input_dict[filename])
            s1_goal_y.append(goal_y[file_count])
            s1_action_y.append(action_y[file_count])
        elif 16 <= int(filename[1:3]) <= 28:
            s2_x.append(input_dict[filename])
            s2_goal_y.append(goal_y[file_count])
            s2_action_y.append(action_y[file_count])
        elif 29 <= int(filename[1:3]) <= 41:
            s3_x.append(input_dict[filename])
            s3_goal_y.append(goal_y[file_count])
            s3_action_y.append(action_y[file_count])
        elif 42 <= int(filename[1:3]) <= 54:
            s4_x.append(input_dict[filename])
            s4_goal_y.append(goal_y[file_count])
            s4_action_y.append(action_y[file_count])
        file_count += 1
    splits_x = [s1_x, s2_x, s3_x, s4_x]
    splits_goal_y = [s1_goal_y, s2_goal_y, s3_goal_y, s4_goal_y]
    splits_action_y = [s1_action_y, s2_action_y, s3_action_y, s4_action_y]

    final_action_acc = final_goal_acc = 0
    for i in range(4):
        trainX = []
        trainY_goal = []
        trainY_action = []
        for j in range(4):
            if i != j:
                trainX += copy.deepcopy(splits_x[j])
                trainY_goal += copy.deepcopy(splits_goal_y[j])
                trainY_action += copy.deepcopy(splits_action_y[j])
        trainY = [trainY_action, trainY_goal]
        testX = copy.deepcopy(splits_x[i])
        testY_goal = copy.deepcopy(splits_goal_y[i])
        testY_action = copy.deepcopy(splits_action_y[i])
        testY = [testY_action, testY_goal]
        sample_weight = get_sample_weight(trainX)

        # train and evaluate the model
        train_model(trainX, trainY, model, sample_weight)
        action_acc, goal_acc = evaluation(testX, testY, model, max_timesteps)

        final_action_acc += action_acc
        final_goal_acc += goal_acc
    #         print(final_acc)
    final_action_acc = final_action_acc / 4
    final_goal_acc = final_goal_acc/4
    return final_action_acc, final_goal_acc


def display_acc(results, model_name, w_or_wo):
    # loop through result for each input proportion

    print("Model: ", model_name, "     With_Activity: ", w_or_wo, "        action accuracy")
    for proportion, acc in results.items():
        #    print("Input(%): ", proportion * 100)
        print(round(acc[0], 5), " ", end="")
    print("")
    print("Model: ", model_name, "     With_Activity: ", w_or_wo, "       goal")
    for proportion, acc in results.items():
        print(round(acc[1], 5), " ", end="")
    print("")

def run_model():
    data_dict = import_data()
    data_dict = cut_zeros(data_dict)
    max_action_len = find_max_action_length(data_dict)
    action_label = get_action_label()
    wo_lstm = lstm_model(max_action_len)
    # wo_tcn = tcn_model(max_action_len)
    # wo_transformer = transformer_model(max_action_len)


    wo_lstm_results = {}
    # w_lstm_results = {}
    wo_tcn_results = {}
    wo_transformer_results = {}
    listt = []
    # get input data with different proportion
    # input_proportion = [0.1]
    input_proportion = [0.1, 0.2, 0.3, 0.4, 0.5]
    # input_proportion = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for proportion in input_proportion:
        #     input_dict = get_input_x(data_dict, proportion)
        #     train_y = get_input_y(data_dict, input_dict)
        input_dict, train_goal_y, train_action_y = get_input_x_y(data_dict, proportion, 2)
        input_dict = frames_to_action(input_dict)
        input_dict = add_padding_to_x(input_dict, max_action_len)
        # train_action_y= add_padding_to_y(train_action_y, max_action_len)
        encoded_goal_y = np.array(label_encoding(train_goal_y, action_label, 1))
        encoded_action_y = np.array(label_encoding(train_action_y, action_label, 0))
        wo_activity_input = feature_encoding(input_dict, action_label)


        # # # lstm
        wo_lstm_action_acc, wo_lstm_goal_acc = cross_validation(wo_activity_input, encoded_goal_y, encoded_action_y,
                                                                wo_lstm, max_action_len, 1)
        wo_lstm_results[proportion] = [wo_lstm_action_acc, wo_lstm_goal_acc]
        listt.append(wo_lstm_action_acc)
        # # tcn

        # wo_tcn_action_acc, wo_tcn_goal_acc = cross_validation(wo_activity_input, encoded_goal_y, encoded_action_y,
        #                                                       wo_tcn, max_action_len, 1)
        # wo_tcn_results[proportion] = [wo_tcn_action_acc, wo_tcn_goal_acc]

        # # transformer
        # wo_transformer_action_acc, wo_transformer_goal_acc = cross_validation(wo_activity_input, encoded_goal_y,
        #                                                                       encoded_action_y,
        #                                                                       wo_transformer, max_action_len, 1)
        # wo_transformer_results[proportion] = [wo_transformer_action_acc, wo_transformer_goal_acc]

    wo_sd = statistics.stdev(wo_lstm_results)
    print("wo activity sd: ", wo_sd)
    #
    # display_acc(wo_lstm_results, "LSTM", "No")
    # display_acc(wo_tcn_results, "TCN", "No")
    display_acc(wo_transformer_results, "Transformer", "No")


run_model()
