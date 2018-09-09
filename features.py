
# coding: utf-8

# In[1]:


# coding: utf-8

import pandas as pd
import numpy as np
import os
from scipy.stats import entropy

def get_mean(signal):
    """
    Get the mean of signal

    Arguments:
    signal -- the original signal

    Return:
    ans -- mean of signal
    """
    arr = np.array(signal)
    return np.mean(arr.astype(np.float))

def get_RMS(signal):
    """
    Get the rms of signal

    Arguments:
    signal -- the original signal

    Return:
    ans -- rms of signal
    """
    ans = np.sqrt(np.mean(signal**2))
    return ans
def get_ZC(signal):
    """
    Get the ratio of zero cross rate

    Arguments:
    signal -- the original signal

    Return:
    ans -- the ratio of zero cross rate
    """
    count = 0
    for i in range(len(signal)-1):
        if (signal[i]*signal[i+1])<0:
            count = count+1
    ans = count/(len(signal)-1)
    return ans
def get_kurt(signal):
    """
    Get the kurt of signal

    Arguments:
    signal -- the original signal

    Return:
    ans -- the kurt of signal
    """
    mean = get_mean(signal)
    m4 = np.mean((signal-mean)**4)
    m2 = (np.mean((signal-mean)**2))**2
    ans = (m4/m2)-3
    return ans

def get_skew(signal):
    """
    Get the skew of signal

    Arguments:
    signal -- the original signal

    Return:
    ans -- the skew of signal
    """
    mean = get_mean(signal)
    m4 = np.mean((signal-mean)**4)
    m2 = (np.mean((signal-mean)**2))**(3/2)
    ans = m4/m2
    return ans

def get_sma_numpy(signal):
    """
    Get the signal magnitude area:
    measure of the magnitude of a varying quantity.

    Arguments:
    signal -- the original signal

    Return:
    get_sum/len(signal) -- the statistical value
    """
    ans = 0
    if signal.shape[1] == 3:
        for i in range (len(signal)):
            ans += (abs(signal[i,0])+abs(signal[i,1])+abs(signal[i,2]))
    elif signal.shape[1] == 8:
        for i in range (len(signal)):
            ans += (abs(signal[i,0])+abs(signal[i,1])+abs(signal[i,2])+abs(signal[i,3])
                    +abs(signal[i,4])+abs(signal[i,5])+abs(signal[i,6])+abs(signal[i,7]))
    else:
        print('The dimension of the input is incorrect')
    return ans/len(signal)

def get_entropy(signal):
    """
    Get the entropy of signal

    Arguments:
    signal -- the original signal

    Return:
    ans -- the entropy of signal
    """

    signal_normalized = signal/max(abs(signal))
    ans = entropy(abs(signal_normalized))
    return ans

def get_rising_time(signal):
    """
    Get the rising time from 10% of largest value of signal to 90% of largest value of signal

    Arguments:
    signal -- the original signal

    Return:
    ans -- the rising time from 10% of largest value of signal to 90% of largest value of signal
    """

    #get the 10% and 90% of maximal value of signal
    maxamp = get_max_amp(signal)
    up = 0.9*maxamp
    low = 0.1*maxamp

    #indicator for finding the lower and upper bound
    findlow = False
    findup = False

    for i in range(len(signal)):

        #if lower/upper bound not found, and we meet the first value the exceed the bound, store it and inverse the flag
        if  (findlow==False) & (signal.iloc[i].values[0]>low):
            t1 = i

            findlow=True
        if (findup==False) & (signal.iloc[i].values[0]>up):
            t2 = i

            findup = True
        if findlow & findup:
            ans = np.float(t2-t1)
            return ans #should multiply by freq: eda=4,bvp=64
        
def get_energy(signal):
    """
    Get the energy value of signal

    Arguments:
    signal -- the original signal

    Return:
    ans -- energy value of signal
    """
    ans = sum([x**2 for x in signal])
    return ans

def get_max_amp(signal):
    """
    Get the maximal value of signal

    Arguments:
    signal -- the original signal

    Return:
    ans -- maximal value of signal
    """
    ans = signal.values.max()
    return ans

def get_std(signal):
    """
    Get the std of signal

    Arguments:
    signal -- the original signal

    Return:
    ans -- std of signal
    """
    ans = np.std(signal)
    return ans

def first_order_diff(X):
    """ Compute the first order difference of a time series.

        For a time series X = [x(1), x(2), ... , x(N)], its	first order
        difference is:
        Y = [x(2) - x(1) , x(3) - x(2), ..., x(N) - x(N-1)]

    """
    D=[]

    for i in range(1,len(X)):
        D.append(X[i]-X[i-1])

    return D

def get_pfd(X):
    """Compute Petrosian Fractal Dimension of a time series from either two
    cases below:
        1. X, the time series of type list (default)
        2. D, the first order differential sequence of X (if D is provided,
           recommended to speed up)

    In case 1, D is computed by first_order_diff(X) function of pyeeg

    To speed up, it is recommended to compute D before calling this function
    because D may also be used by other functions whereas computing it here
    again will slow down.
    """
    D = None
    if D is None:
        D = first_order_diff(X)
    N_delta= 0; #number of sign changes in derivative of the signal
    for i in range(1,len(D)):
        if D[i]*D[i-1]<0:
            N_delta += 1
    n = len(X)
    return np.log10(n)/(np.log10(n)+np.log10(n/n+0.4*N_delta))

def get_bin_power(X, Band):
    Fs = 50
    C = abs(X)
    Power = np.zeros(len(Band)-1);
    for Freq_Index in range(0,len(Band)-1):
        Freq = float(Band[Freq_Index])
        Next_Freq = float(Band[Freq_Index+1])
        Power = sum(C[int(np.floor(Freq/Fs*len(X))):int(np.floor(Next_Freq/Fs*len(X)))])
    return Power

