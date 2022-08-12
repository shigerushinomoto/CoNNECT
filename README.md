# A python code of CoNNECT: Convolutional Neural Network for Estimating synaptic Connectivity from spike Trains.

## Preparation

Install cython and tensorflow at your PC if you have not done before.

***cython***

```
$ pip install cython
```
We have confirmed that our algorithm runs with the cython version 0.29.6.

***tensorflow***

```
$ pip install tensorflow
```
We have confirmed that our algorithm runs with the tensorflow version 1.13.1.


Then, run setup.py
```
$ cd modules
$ python setup.py build_ext -i
```

## Estimate connectivity with "estimate.py"

```
$ python estimate.py
```
Replace "spikefile" and "savefile" with the file names of your spike train data and estimated PSP data, respectively.
The estimated PSP data is given as a matrix, whose column and row correspond to presynaptic and postsynaptic indices, respectively.

# License
This software is released under the MIT License, see LICENSE.txt.

# About this program
Python program contributed by Daisuke Endo (daisuke.endo96@gmail.com).

Date of the final revision: 2020/05/17

The analysis was directed by Shigeru Shinomoto (shigerushinomoto@gmail.com).

# Reference
D. Endo, R. Kobayashi, R. Bartolo, B.B. Averbeck, Y. Sugase-Miyamoto, K. Hayashi, K. Kawano, B.J. Richmond, and S. Shinomoto, CoNNECT: Convolutional Neural Network for Estimating synaptic Connectivity from spike Trains. bioRxiv 2020.05.05.078089
