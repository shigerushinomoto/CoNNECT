#
# utils.py
# Copyright (c) 2020 Daisuke Endo
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
#
#----------------------------------
# 作成者：Daisuke Endo
# 連絡先:daisuke.endo96@gmail.com
# 最終更新日　2020/5/17
#----------------------------------
# ここには3つの関数を記述している。
# 1. ファイルの読み込み関数
# 2. CrossCorrelogramからmi_sだけ削る関数
# 3. 結合行列に変換する関数
#----------------------------------

import os
import numpy as np

def load_data(filepath):
    """
    ;区切りのデータをnumpyで扱える形に読み込む

    args:
        filepath: ファイル名
    return
        list形式でそれぞれのニューロンのspike列(ndarray)を入れたもの。
    """
    with open(filepath, "r") as f:
        lines = f.read()
        lines = lines.split(";")
    N_neuron = len(lines) - 1 # 最後は\nだけのデータのため, １を引く

    # list形式でspike列を格納する
    spikes = []
    for index in range(N_neuron):
        spikes.append(np.array([x for x in lines[index].split("\n") if x],
                               dtype=np.float64))

    return spikes

def pickup_bin(hist, begin0, end0, begin1, end1):
    """
    begin0 ~ end0, begin1 ~ end1の部分のCCだけ取り出す。
    ここで、begin0等の値は、中間の値を0msだと思った時の値である。
    つまり、-52 ~ -2, 2　~ 52など
    args:
        hist: ndarray hist.shape = (データ数, histogramの範囲)
        begin0, end0, begin1, end1: int型
    return:
        histから必要な範囲のhistogramを取り出したndarray
    """
    center = int(hist.shape[1]/2)

    # shift
    begin0 += center
    end0 += center
    begin1 += center
    end1 += center

    return np.hstack([hist[:, begin0:end0], hist[:, begin1:end1]])

def toPSPmatrix(psp, N_neuron, index):
    """
    推定したPSPの値からPSP matrix作る
    indexをもとに、行列を作成する。
    args:
        psp: 推定結果のndarray
    return:
        psp_mat: PSP行列の2次元ndarray。行は結合先、列は結合元のニューロン番号を表す。
    """
    # hintom diagramにする
    psp_mat = np.zeros([N_neuron, N_neuron])

    # 明示的にindexが与えられているのならば
    assert len(psp) == len(index), "Error: It's nessesary that len(psp) == len(index), but {} != {}".format(len(psp), len(index))

    # 推定された結合を行列に変形する
    N_conn = len(psp)
    for i in range(N_conn):
        post = index[i][0]
        pre = index[i][1]

        # psp_matに入れていく
        psp_mat[post, pre] = psp[i]

    return psp_mat
