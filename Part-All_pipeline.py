import pandas as pd
import numpy as np
import pickle
import os
import multiprocessing as mp
import statsmodels.api as sm
from argparse import ArgumentParser, RawTextHelpFormatter
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn import utils
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from pydeseq2.dds import DeseqDataSet
from argparse import ArgumentParser, RawTextHelpFormatter
from datetime import datetime
from lifelines import CoxPHFitter


# 参数
def __init__():
    parser = ArgumentParser(
        formatter_class = RawTextHelpFormatter,
        description = 'Identify marker proteins.'
    )
    parser.add_argument(
        '-v', '--version', action = 'version', version = '%(prog)s 1.0.0'
    )
    parser.add_argument(
        '-gtex', '--GTEx-data', type = str, required = True, metavar = '<str>',
        help = 'Path to the GTEx matrix file.'
    )
    parser.add_argument(
        '-pro', '--proteins-matrix', type = str, required = True, metavar = '<str>',
        help = 'Path to the proteins matrix file.'
    )
    parser.add_argument(
        '-organ', '--organ-fc', default = 2.0, type = float, required = False, metavar = '<float>',
        help = 'Organ marker identify foldchange.\nDefault: 2.0'
    )
    parser.add_argument(
        '-brain', '--brain-fc', default = 1.5, type = float, required = False, metavar = '<float>',
        help = 'Brain region marker identify foldchange.\nDefault: 1.5'
    )
    parser.add_argument(
        '-marker', '--marker-list', type = str, required = True, metavar = '<str>',
        help = 'Path to the brain region marker proteins list file.'
    )
    parser.add_argument(
        '-r', '--pearson-r', type = float, default = 0.05, required = False, metavar = '<float>',
        help = 'Filter models with pearsonr value.\nDefault: 0.05'
    )
    parser.add_argument(
        '-nbootstrap', '--bootstrap-times', type = int, default = 100, required = False, metavar = '<int>',
        help = 'Bootstrap training times.\nDefault: 100'
    )
    parser.add_argument(
        '-k', '--k-fold', default = 5, type = int, required = False, metavar = '<int>',
        help = 'K-fold training.\nDefault: 5'
    )
    parser.add_argument(
        '-t', '--threads-use', default = 2, type = int, required = False, metavar = '<int>',
        help = 'Threads numbers to use.\nDefault: 2'
    )
    parser.add_argument(
        '-disease', '--disease-pro', type = str, required = True, metavar = '<str>',
        help = 'Path to the disease proteins matrix directory.'
    )
    parser.add_argument(
        '-p', '--p-value', default = 0.05, type = float, required = False, metavar = '<float>',
        help = 'P-value filtered for associations.\nDefault: 0.05'
    )
    parser.add_argument(
        '-o', '--output', type = str, required = True, metavar = '<str>',
        help = 'Path to the output file.'
    )
    return parser.parse_args()