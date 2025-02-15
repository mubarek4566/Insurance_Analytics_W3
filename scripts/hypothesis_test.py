import pandas as pd
from scipy.stats import f_oneway, ttest_ind, chi2_contingency, chi2
from scipy import stats

class HypothessTest:
    def __init__(self):
        self.df = {}
    
    