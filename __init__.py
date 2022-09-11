'''
Created on Aug 7, 2022

@author: Faizan3800X-Uni
'''
import os
from multiprocessing import current_process

# Due to shitty tkinter errors.
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# Numpy sneakily uses multiple threads sometimes. I don't want that.
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPI_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

from .core import IAAFTSAMain, IAAFTSAPlot

current_process().authkey = 'phsann'.encode(encoding='utf_8', errors='strict')
