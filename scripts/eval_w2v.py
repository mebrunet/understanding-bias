import sys
from time import time
from os import path
import json

import numpy as np
from numpy import random as npr
from gensim.models.word2vec import Word2Vec


TEST_MODEL_PATH = '/Users/mebrunet/Code/UofT/understanding-bias/results/w2v/nyt_baseline_200-8-15-1.w2v'
TEST_WORDS_PATH = '/Users/mebrunet/Code/UofT/understanding-bias/scripts/question-words.txt'
WEAT_1_words = {"S": ("science", "technology", "physics", "chemistry",
                      "einstein", "nasa", "experiment", "astronomy"),
                "T": ("poetry", "art", "shakespeare", "dance", "literature",
                      "novel", "symphony", "drama"),
                "A": ("male", "man", "boy", "brother", "he", "him", "his",
                      "son"),
                "B": ("female", "woman", "girl", "sister", "she", "her",
                      "hers", "daughter")}


def get_arg(i, default=None):
    '''Helper to get command line arg or return a default'''
    arg = sys.argv[i] if len(sys.argv) > i else default
    if arg is None:
        raise TypeError('Missing required argument (position {}).'.format(i))
    return arg


def percent_accuracy(acc):
    return len(acc[-1]['correct']) / (len(acc[-1]['correct']) + len(acc[-1]['incorrect']))


#helper function to calculate S(w_word, A_att, B_att)
def calc_w_stat(w_word, A_att, B_att, my_model):
    try:
        my_model.word_vec(w_word)
    except KeyError:
       # print("Error - word ", w_word," not found")
        return np.inf

    my_stat_wA = 0
    my_stat_wB = 0
    num_errors = 0

    for a in range(0, len(A_att)):
        #find cosine simliarity between words
        try:
            my_stat_wA = my_stat_wA + my_model.similarity(w_word, A_att[a])
        except KeyError:
           # print("Error - word ", A_att[a] , " not found")
            num_errors +=1

    my_stat_wA = np.divide(my_stat_wA, len(A_att) - num_errors) #average cosine similarities

    num_errors =0 #reset number of errors
    for b in range(0, len(B_att)):
        try:
            my_stat_wB = my_stat_wB + my_model.similarity(w_word, B_att[b])
        except KeyError:
           # print("Error - word ", B_att[b] , " not found")
            num_errors +=1

    my_stat_wB = np.divide(my_stat_wB, len(B_att) - num_errors) #average cosine similarities

    my_stat_w = my_stat_wA - my_stat_wB
    return my_stat_w


def calc_effect_size (X_tar, Y_tar, A_att, B_att, my_model):
    my_stat_X = 0 # mean over x in X_tar of s(x, A_att, B_att)
    my_stat_Y = 0

    #create array to store s(w, A_att, B_att) for standard dev. calc later
    stat_array = np.zeros(len(X_tar)+len(Y_tar))
    countx = 0
    county = 0

    for x in range(0, len(X_tar)):
        word_x = calc_w_stat(X_tar[x], A_att, B_att, my_model)

        if word_x != np.inf:
            my_stat_X = my_stat_X + word_x
            stat_array[countx] = word_x
            countx +=1

    my_stat_X = np.divide(my_stat_X, countx) #finds mean

    for y in range(0, len(Y_tar)):
        word_y = calc_w_stat(Y_tar[y], A_att, B_att, my_model)

        if word_y != np.inf:
            my_stat_Y = my_stat_Y + word_y
            stat_array[countx + county] = word_y
            county +=1

    my_stat_Y = np.divide(my_stat_Y, county) #finds mean

    stat_array = stat_array[:countx + county -1]

    st_dev = np.std(stat_array)

    effect_size = np.divide(my_stat_X - my_stat_Y, st_dev)
    return effect_size

if __name__ == '__main__':
    model_path = get_arg(1, TEST_MODEL_PATH)
    test_words_path = get_arg(2, TEST_WORDS_PATH)
    filename = path.basename(model_path)
    scenario = filename.split('_')[1]
    intensity = filename.split('_')[2]
    seed = filename[0:-4].split('-')[-1]
    w2v = Word2Vec.load(model_path)
    acc_report = w2v.wv.accuracy(test_words_path)
    acc = percent_accuracy(acc_report)
    es = calc_effect_size(WEAT_1_words["S"], WEAT_1_words["T"],
                          WEAT_1_words["A"], WEAT_1_words["B"], w2v.wv)
    print(",".join([scenario, intensity, seed, str(w2v.corpus_count), str(acc), str(es)]))
