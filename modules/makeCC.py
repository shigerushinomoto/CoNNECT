#
# makeCC.py
# Copyright (c) 2020 Daisuke Endo
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
#
#----------------------------------
# 作成者：Daisuke Endo
# 連絡先:daisuke.endo96@gmail.com
# 最終更新日　2020/5/17
#----------------------------------
# ここには2つの関数を記述している。
# 1. あるペアのCrossCorrelogramを計算する関数
# 2. 全てのペアのCrossCorrelogramを計算する関数
#----------------------------------

import cython
import numpy as np
import math
import sys
from joblib import Parallel, delayed
from crosscorrelogram import computeCC, computeAutoCC

def computeHist(i, j, spiketrain, Begin, End):
    """
    jから見た時のiの相対時刻を求めて、CrossCorrelogramを計算する。
    """
    spike0 = spiketrain[i]
    spike1 = spiketrain[j]
    histogram = computeCC(spike1, spike0, Begin, End)
    return i, j, histogram

def makeCC_allpair(spikes, Begin, End, N_thred):
    """
    全てのCCをBegin~Endの間で計算する
    args:
        spikes: list型
        Begin, End: int型
        N_thred: 並列計算をするときに与えるスレッド数
    return:
        X: ペアごとに計算したCC. X.shape = (ニューロンのペア数、CCの幅)
        index: ペアのニューロンの番号. index.shape = (ニューロンのペア数、2). index[i][0]は結合先ニューロン、index[i][1]は結合元のニューロンを表す.
    """
    #-----------------------------------------------------
    def func(i, j, spiketrain, Begin, End):
        """
        並列計算の中に入れる関数.
        Auto CrossCorrelogramを計算する場合と別のニューロン間のCrossCorrelogramで分けている.
        """
        if i > j:
            pass
            # print("Because {} > {}, we pass this pair".format(i, j))
        elif i == j:
            # print("Because {} == {}, we compute autoCC".format(i, j))
            histogram = computeAutoCC(spiketrain[i], Begin, End)
            return i, j, histogram
        else:
            return computeHist(i, j, spiketrain, Begin, End)
    #-----------------------------------------------------
    # 並列計算でCCを計算する
    N_neuron = len(spikes)
    print("Compute Cross-Correlogram between {} ms and {} ms".format(Begin, End))

    result = Parallel(n_jobs=N_thred)([delayed(func)(i, j, spikes, Begin, End) for i in range(N_neuron) for j in range(N_neuron)])

    # 計算結果を扱いやすい形に変更する
    X = []
    index = []
    for _ in range(len(result)):
        # 計算していないデータ(i<j)はNoneが入っているのでpassする
        if result[_] == None:
            pass
        else:
            # i, jのこと. neuronのindexに変換する
            post = result[_][0]
            pre = result[_][1]
            cc = result[_][2]

            if post == pre:
                # i == jについて
                index.append([post, pre])
                X.append(cc)

            else:
                # i < jについて
                index.append([post, pre])
                X.append(cc)

                # i > j. 反転した方向も作る
                index.append([pre, post])
                X.append(cc[::-1])

    return np.array(X), np.array(index)
