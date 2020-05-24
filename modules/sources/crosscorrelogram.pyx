#
# crosscorrelogram.pyx
# Copyright (c) 2020 Daisuke Endo
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
#
#----------------------------------
# 作成者：Daisuke Endo
# 連絡先:daisuke.endo96@gmail.com
# 最終更新日　2020/5/17
#----------------------------------
# ここではCCを計算する関数をcythonで記述している。
#----------------------------------

import cython
import numpy as np
cimport numpy as cnp
import math

def computeCC(spikeR, spikeT, int Begin=-55, int End=55):
    """
    spikeR[i]からみたspikeT[j]の相対時刻を計算する.
    余計なものも作成されるが-54.999999から54.99999まで取り出す
    Begin, Endぴったしの値は省く.
    イメージ:
        historgram[3]は3ms~3.9999msまでの値を含む

    args:
        spikeR, spikeT: スパイク列
    return:
       histogram: crosscorrelogramのndarray.
    """
    cdef:
        int n_dim, i, j, Start, Stop, N_spike, new_Start
        float delta
        int update
        cnp.ndarray histogram

    n_dim = int(End - Begin)
    # 先に箱を作っておく
    histogram = np.zeros(n_dim, dtype=np.int64)

    # 全てのspikeRについてループする
    # spikeTの探索範囲をStart, Stopとする. これはindex
    Stop = len(spikeT)
    Start = 0
    new_Start = 0
    N_spike = len(spikeR)
    # spikeRについてのループ
    for i in range(N_spike):

        # Start位置の更新済みフラグ
        update = False

        # spikeTについてのループ
        for j in range(Start, Stop):

            # spikeRからみた相対時刻をdeltaとする
            delta = spikeT[j] - spikeR[i]

            # 最初に見つけたBegin以上の値をStartとして更新. Update flgを立てる
            if (not update) and (delta > Begin):
                new_Start = j
                update = True

            # End msを超えたらbreak
            if delta-Begin >= n_dim:
                break

            # historgramに追加していく
            if delta > Begin:
                histogram[math.floor(delta-Begin)] += 1

        # Startの更新.
        Start = new_Start

    return histogram


def computeAutoCC(spike, int Begin=-55, int End=55):
    """
    自己相関の計算を行う.
    余計なものも作成されるが-54.999999から54.99999まで取り出せば良い
    Begin, Endぴったしの値は省くことにする.
    自分と同じ値は省く.

    args:
        spike: スパイク列
    return:
       histogram: crosscorrelogramのndarray.
    """
    cdef:
        int n_dim, i, j, Start, Stop, N_spike, new_Start
        float delta
        int update
        cnp.ndarray histogram

    n_dim = int(End - Begin)
    # 先に箱を作っておく
    histogram = np.zeros(n_dim, dtype=np.int64)

    # 全てのspikeについてループする
    # spikeの探索範囲をStart, Stopとする. これはindex
    Stop = len(spike)
    Start = 0
    new_Start = 0
    N_spike = len(spike)
    # spikeについてのループ
    for i in range(N_spike):

        # Start位置の更新済みフラグ
        update = False

        # spikeについてのループ
        for j in range(Start, Stop):

            # spikeからみた相対時刻をdeltaとする. 同時刻はpassする
            delta = spike[j] - spike[i]

            # 最初に見つけたBegin以上の値をStartとして更新. Update flgを立てる
            if (not update) and (delta > Begin):
                new_Start = j
                update = True

            if delta == 0:
                continue

            # End msを超えたらbreak
            if delta-Begin >= n_dim:
                break

            # historgramに追加していく
            if delta > Begin:
                histogram[math.floor(delta-Begin)] += 1

        # Startの更新.
        Start = new_Start

    return histogram
