#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:51:46 2026

@author: giacomomarocco
"""

import numpy as np

class FeedbackForces:
    ''' This class contains various feedback forces that can be called
    once the state is specified (i.e. xc and pc) '''
    def __init__(self, g_fb, omega = 1.0):
        self.g_fb = g_fb
        self.omega = omega
    def momentum_feedback(self):
        '''Damp the momentum'''
        return lambda xc, pc: - np.sqrt(2) * self.g_fb * pc 
    def optimal_feedback(self):
        '''Do the LQG optimum, as per my notes'''
        q = self.omega / self.g_fb**2
        r = self.omega * q
        alpha = np.sqrt(1 + 1/r)
        gamma_x = alpha - 1
        gamma_p = np.sqrt(1/r - 2 + 2 * alpha)
        return lambda xc, pc: - gamma_x * xc - gamma_p * pc 

    