# todo: ignore video if there is no next action
# todo: model3, output both goal and next action
# import
from sklearn import metrics
import scipy.io
import os
import numpy as np
import copy
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


def get_input_x_y(data_dict, trainPortion):
    new_dict = {}
    train_y = []

    for filename, frames in data_dict.items():
        has_next = 0
        frameLen = len(frames)
        inputLen = round(frameLen * trainPortion)
        inputFrames = frames[:inputLen]
        y = [frames[inputLen - 1][0]]
        for frame in frames[inputLen:]:
            #             print(frame, y)
            if frame[0] != y[0]:
                has_next = 1
                y[0] = frame[0]

                break

        if has_next:
            train_y.append(y)
            new_dict[filename] = inputFrames
    return new_dict, train_y


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


def get_activity(filename):
    # convert filename to activities
    activity = (filename.split("_", 3)[-1]).split(".")[0]
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



def label_encoding(trainY, action_label):
    new_trainY = []
    for frames in trainY:
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


def lstm_model(max_action_len, with_activity):
    if not with_activity:
        data_dimension = 47
    else:
        data_dimension = 57
    model = Sequential()
    model.add(LSTM(100, input_shape=(max_action_len, data_dimension), return_sequences=False))
    model.add(Dense(47, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model



def tcn_model(max_action_len, with_activity):
    if not with_activity:
        data_dimension = 47
    else:
        data_dimension = 57
    i = Input(shape=(max_action_len, data_dimension))
    o = TCN(128, return_sequences=False)(i)  # The TCN layers are here.
    o = TCN(64, return_sequences=False)(i)
    o = Dense(47, activation = "softmax")(o)
    model = Model(inputs=[i], outputs=[o])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

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

def transformer_model(max_action_len, with_activity):
    if not with_activity:
        data_dimension = 47
    else:
        data_dimension = 57
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

    inputs = layers.Input(shape=(max_action_len, data_dimension))
    transformer_block = TransformerBlock(data_dimension, 1, ff_dim)
    x = transformer_block(inputs)
    # x = layers.GlobalAveragePooling1D()(x)
    # x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation="relu")(x)
    # # x = layers.Dense(64, activation="relu")(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(47, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'],
                  sample_weight_mode="temporal")
    # sample_weight_mode = "temporal"
    model.summary()
    return model

# cross validation
def train_model(trainX, trainY, model, sample_weight):
    print(np.array(trainX).shape, np.array(trainY).shape)
    model.fit(np.array(trainX), np.array(trainY), epochs=20, batch_size=128)


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
    #     print(predictions)
    results = []
    accuracy_list = []
    for i in range(len(predictions)):
        each_video = []
        result = np.array([0] * 47)
        index = predictions[i].argmax(axis=-1)
        result[index] = 1
        results.append(result)


    correct = 0
    valid = 0
    # loop each video
    for j in range(len(testY)):

        if not (results[j] == 47 * [0]).all():
            valid += 1
            if (testY[j] == results[j]).all():
                correct += 1
    if valid == 0:
        accuracy_list.append(0)
        print("timestep", i + 1, ":", 0)
    else:
        accuracy_list.append(correct / valid)
        print("timestep", i + 1, ":", correct / valid)
    return correct / valid



def cross_validation(input_dict, encoded_y, model, max_timesteps):
    file_count = 0
    s1_x, s2_x, s3_x, s4_x = [], [], [], []
    s1_y, s2_y, s3_y, s4_y = [], [], [], []
    # s1: P03 – P15
    # s2: P16 – P28
    # s3: P29 – P41
    # s4: P42 – P54
    count = 0
    print(len(input_dict))

    for filename, frames in input_dict.items():
        if int(filename[1:3]) <= 15:
            s1_x.append(input_dict[filename])
            s1_y.append(encoded_y[file_count])
        elif 16 <= int(filename[1:3]) <= 28:
            s2_x.append(input_dict[filename])
            s2_y.append(encoded_y[file_count])
        elif 29 <= int(filename[1:3]) <= 41:
            s3_x.append(input_dict[filename])
            s3_y.append(encoded_y[file_count])
        elif 42 <= int(filename[1:3]) <= 54:
            s4_x.append(input_dict[filename])
            s4_y.append(encoded_y[file_count])
        file_count += 1
    splits_x = [s1_x,s2_x,s3_x,s4_x]
    splits_y = [s1_y,s2_y,s3_y,s4_y]
    # splits_x = [np.array(s1_x), np.array(s2_x), np.array(s3_x), np.array(s4_x)]
    # splits_y = [np.array(s1_y), np.array(s2_y), np.array(s3_y), np.array(s4_y)]
    final_acc = []

    for i in range(4):
        trainX = None
        trainY = None
        for j in range(4):
            if i != j:
                if trainX == None:
                    trainX = copy.deepcopy(splits_x[j])
                else:
                    trainX += copy.deepcopy(splits_x[j])
                if trainY == None:
                    trainY = copy.deepcopy(splits_y[j])
                else:
                    trainY += copy.deepcopy(splits_y[j])
        testX = copy.deepcopy(splits_x[i])
        testY = copy.deepcopy(splits_y[i])
        #         print(np.array(trainX).shape, np.array(trainY).shape,)
        sample_weight = get_sample_weight(trainX)
        train_model(trainX, trainY, model, sample_weight)
        if final_acc == []:
            final_acc = evaluation(testX, testY, model, max_timesteps)
        else:
            final_acc += evaluation(testX, testY, model, max_timesteps)
    #         print(final_acc)
    final_acc = final_acc / 4
    return (final_acc)


def display_acc(results, model_name, w_or_wo):
    # loop through result for each input proportion

    print("Model: ", model_name, "     With_Activity: ", w_or_wo)
    for proportion, acc in results.items():
    #    print("Input(%): ", proportion * 100)
        print(round(acc, 5)," ", end="")
    print("")



def run_model():
    data_dict = import_data()
    data_dict = cut_zeros(data_dict)
    max_action_len = find_max_action_length(data_dict)
    action_label = get_action_label()
    wo_lstm = lstm_model(max_action_len, 0)
    w_lstm = lstm_model(max_action_len, 1)
    wo_tcn = tcn_model(max_action_len, 0)
    w_tcn = tcn_model(max_action_len, 1)
    wo_transformer = transformer_model(max_action_len, 0)
    w_transformer = transformer_model(max_action_len, 1)

    wo_lstm_results = {}
    w_lstm_results = {}
    wo_tcn_results = {}
    w_tcn_results = {}
    wo_transformer_results = {}
    w_transformer_results = {}
    # get input data with different proportion
    # input_proportion = [0.1]
    # input_proportion = [0.1, 0.2, 0.3, 0.4, 0.5]
    input_proportion = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for proportion in input_proportion:
    #     input_dict = get_input_x(data_dict, proportion)
    #     train_y = get_input_y(data_dict, input_dict)
        input_dict, train_y = get_input_x_y(data_dict, proportion)
        input_dict = frames_to_action(input_dict)
        input_dict = add_padding_to_x(input_dict, max_action_len)
        # train_y = add_padding_to_y(train_y, max_action_len)
        encoded_y = np.array(label_encoding(train_y, action_label))
        wo_activity_input = feature_encoding(input_dict, action_label)
        w_activity_input = add_activity(wo_activity_input)

        # # # lstm
        # wo_lstm_result = cross_validation(wo_activity_input, encoded_y, wo_lstm, max_action_len)
        # wo_lstm_results[proportion] = wo_lstm_result
        # w_lstm_result = cross_validation(w_activity_input, encoded_y, w_lstm, max_action_len)
        # w_lstm_results[proportion] = w_lstm_result
        #
        # # tcn
        # wo_tcn_result = cross_validation(wo_activity_input, encoded_y, wo_tcn, max_action_len)
        # wo_tcn_results[proportion] = wo_tcn_result
        # w_tcn_result = cross_validation(w_activity_input, encoded_y, w_tcn, max_action_len)
        # w_tcn_results[proportion] = w_tcn_result

        # transformer
        wo_transformer_result = cross_validation(wo_activity_input, encoded_y, wo_transformer, max_action_len)
        wo_transformer_results[proportion] = wo_transformer_result
        w_transformer_result = cross_validation(w_activity_input, encoded_y, w_transformer, max_action_len)
        w_transformer_results[proportion] = w_transformer_result
    #
    # display_acc(wo_lstm_results, "LSTM", "No")
    # display_acc(w_lstm_results, "LSTM", "Yes")
    # display_acc(wo_tcn_results, "TCN", "No")
    # display_acc(w_tcn_results, "TCN", "Yes")
    display_acc(wo_transformer_results, "Transformer", "No")
    display_acc(w_transformer_results, "Transformer", "Yes")
run_model()
