import numpy as np
import sys
import argparse
import glob
import os

def sigma_rel_to_gamma(sigma_rel):
    t = sigma_rel ** -2
    gamma = np.roots([1, 7, 16 - t, 12 - t]). real.max()
    return gamma

def p_dot_p(t_a , gamma_a , t_b , gamma_b ):
    t_ratio = t_a / t_b
    t_exp = np.where(t_a < t_b , gamma_b , -gamma_a)
    t_max = np.maximum(t_a , t_b)
    num = (gamma_a + 1) * (gamma_b + 1) * t_ratio ** t_exp
    den = (gamma_a + gamma_b + 1) * t_max
    return num / den

def solve_weights(t_i , gamma_i , t_r , gamma_r ):
    rv = lambda x: np.float64(x).reshape(-1, 1)
    cv = lambda x: np.float64(x).reshape(1, -1)
    A = p_dot_p(rv(t_i), rv(gamma_i), cv(t_i), cv(gamma_i ))
    B = p_dot_p(rv(t_i), rv(gamma_i), cv(t_r), cv(gamma_r ))
    X = np.linalg.solve(A, B)
    return X

