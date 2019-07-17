#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 12:01:12 2019

@author: emcastro
"""

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

D = 15
typ = "gradient" # gradient or random

def create_R(angle):
    return np.asarray([[np.cos(angle), np.sin(angle)],
                       [-np.sin(angle), np.cos(angle)]])

def compute_coeff(point):
    x, y = point

    x1 = np.floor(x)
    x2 = x1+1

    y1 = np.floor(y)
    y2 = y1+1

    a = (x2-x)*(y2-y)
    b = (x-x1)*(y2-y)
    c = (x2-x)*(y-y1)
    d = (x-x1)*(y-y1)
    return a, b, c, d

def compute_coeff_coords(point, major):
    x, y = point

    x1 = int(np.floor(x))
    x2 = x1+1

    y1 = int(np.floor(y))
    y2 = y1+1

    if x1 < 0:
        x1 = 0
    if x1 > major:
        x1 = major

    if x2 < 0:
        x2 = 0
    if x2 > major:
        x2 = major

    if y1 < 0:
        y1 = 0
    if y1 > major:
        y1 = major

    if y2 < 0:
        y2 = 0
    if y2 > major:
        y2 = major

    a = [x1, y1]
    b = [x2, y1]
    c = [x1, y2]
    d = [x2, y2]
    return a, b, c, d

def rot_mat(size=(4, 4), angle=0):

    half = (size[0]-1)/2
    xx, yy = np.meshgrid(np.arange(-half, half+1), np.arange(-half, half+1))
    points = np.stack([yy, xx], axis=2)
    points = np.reshape(points, [-1, 2])

    R = create_R(angle)
    points = np.matmul(points, R)

    points = points.reshape((*size, 2))
    points += half
    coeffs = np.zeros([*size, *size])
    for i in range(size[0]):
        for j in range(size[1]):
            c = compute_coeff(points[i, j])
            coords = compute_coeff_coords(points[i, j], major=int(half*2))
            for k in range(4):
                coeffs[i, j, coords[k][0], coords[k][1]] += c[k]
    return coeffs

def custom(shape, axis):
    k = np.zeros(shape)
    p = [(j,i) for j in range(shape[0])
           for i in range(shape[1])
           if not (i == (shape[1] -1)/2. and j == (shape[0] -1)/2.)]

    for j, i in p:
        j_ = int(j - (shape[0] -1)/2.)
        i_ = int(i - (shape[1] -1)/2.)
        k[j,i] = (i_ if axis==0 else j_)/float(i_*i_ + j_*j_)
    return k

def random(shape):
    return np.random.rand(*shape)

if typ == "gradient":
    W = custom((D, D), axis=1)

elif typ == "random":
    W = random((D, D))

W = np.reshape(W, [D, D, 1, 1]).astype(float)

counter = 0
for angle_D in np.linspace(0, 90, 9):
    counter+=1
    angle = np.pi * angle_D / 180
    a = tf.einsum("ijkl,klmn->ijmn", tf.convert_to_tensor(rot_mat((D, D), angle)),
                  tf.convert_to_tensor(W))[:,:,0,0]
    plt.subplot(1, 9, counter)
    plt.imshow(a, cmap="copper")
    plt.axis("off")

plt.show()