#
# estimate.py
# Copyright (c) 2020 Daisuke Endo
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
#
#----------------------------------
# This is a function for estimating synaptic connectivity.
# if __name__ == "__main__"
# "spikefile": a set of spike trains.
# Replace this with the file name of your spike train data.
# "savefile": a set of estimated PSPs.
# Replace this with the file name of estimated PSPs.
#----------------------------------

import sys
import os
import time
import pickle
import numpy as np
sys.path.append('modules/')
from utils import load_data, pickup_bin, toPSPmatrix
from makeCC import makeCC_allpair

import tensorflow as tf
#
# from tensorflow.keras import backend
# gpuConfig = tf.ConfigProto(allow_soft_placement=True,
#                            gpu_options=tf.GPUOptions(allow_growth=True,
#                                                      visible_device_list="1"))

# sess = tf.Session(config=gpuConfig)
# backend.set_session(sess)
from tensorflow.keras.models import load_model


def estimate(spikefile, savefile, min_s=2, conn_threshold=0.5, N_thred=1):
    """
    Estimate a connection matrix from a set of spike trains.
    args:
    spikefile: a set of spike trains. Spike trains are partitioned with semicolons ";". Each spike train consists of spike times partitioned by return codes.
    min_s: an interval for which a center of cross-correlogram is deleted. One can exclude spike records at an interval of ± min_s　ms in the cross-correlogram because near-synchronous spikes were not detected in the experiment due to the shadowing effect.
    conn_threshold: a threshold with which the presence or absence of connectivity is decided.
    N_thred: The number of threads in the parallel calculation. If the number of neurons is about 50, the computation time might become even longer if you increase the number of threads.
    """
    # loading the spike train data:
    spiketrain = load_data(spikefile)
    N_neuron = len(spiketrain)

    # compute the cross correlogram (CC):
    Begin = -50 - min_s
    End = 50 + min_s
    X, index = makeCC_allpair(spiketrain, Begin, End, N_thred)

    # delete a central part of cross-correlogram:
    print("Delete min_s")
    begin0 = -50 - min_s
    end0 = -min_s
    begin1 = min_s
    end1 = 50 + min_s

    X = pickup_bin(X, begin0, end0, begin1, end1)

    # load a model:
    print("Load a CoNNECT model")
    model = load_model("model/model.h5")

    # estimate the connectivity from the cross-correlogram:
    psp_pred, connectivity = model.predict(X)

    # set connection = 0 if the confidence is smaller than the threshold.
    psp_pred[connectivity < conn_threshold] = 0

    # make up the connection matrix (set the diagonal =0):
    psp_mat = toPSPmatrix(psp_pred, N_neuron, index=index)
    np.fill_diagonal(psp_mat, 0)

    # save the estimated connection matrix:
    print("Save estimated PSP matrix")
    np.savetxt(savefile,
          psp_mat,
          delimiter=",")

if __name__ == "__main__":
    spikefile = "sample/spiketrain.txt"
    savefile = "sample/estimated.csv"

    estimate(spikefile, savefile)

##########################################################################
# Python program contributed by Daisuke Endo (daisuke.endo96@gmail.com).

# Date of the final revision: 2020/05/17

# The analysis was directed by Shigeru Shinomoto (shinomoto.shigeru.6e@kyoto-u.ac.jp).

# (reference) D. Endo, R. Kobayashi, R. Bartolo, B.B. Averbeck, Y. Sugase-Miyamoto, K. Hayashi, K. Kawano, B.J. Richmond, and S. Shinomoto, CoNNECT: Convolutional Neural Network for Estimating synaptic Connectivity from spike Trains. bioRxiv 2020.05.05.078089
