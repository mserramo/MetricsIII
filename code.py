import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import raw data
data = pd.read_csv('census_wage_sob.txt', sep=',', header=None, names=['lwage', 'sob'])

#Create variable exp_age, which is exp(lwage)
data['wage'] = np.exp(data['lwage'])

#1.a. Mean, standard deviation, skewness, and kurtosis of exp(lwage)
mean_exp_wage = round(data['wage'].mean(), 2)
std_exp_wage = round(data['wage'].std(), 2)
skew_exp_wage = round(data['wage'].skew(), 2)
kurt_exp_wage = round(data['wage'].kurt(), 2)

#Print the results
print(f'Mean of wage): {mean_exp_wage}')
print(f'Standard deviation of wage: {std_exp_wage}')
print(f'Skewness of wage: {skew_exp_wage}')
print(f'Kurtosis of wage: {kurt_exp_wage}')

#1.b. 
#Compute the inverse of the standard normal distribution CDF 
from scipy.stats import norm

def required_sample_size(alpha, beta, t0, sigma, effect_size_std=None):
    z_alpha = norm.ppf(1- alpha/2)
    z_beta = norm.ppf(beta)
    if effect_size_std is None: 
        n = ((z_alpha + z_beta)**2) / ((t0**2/sigma**2))
    else:
        n = ((z_alpha + z_beta)**2) / ((effect_size_std)**2)
    return n

std_exp_lwage = data['lwage'].std()
min_n_v1 = required_sample_size(0.05, 0.8, 0.04, std_exp_lwage)
print(f'Minimum sample size for version 1: {min_n_v1}') 

#Alternative: using the formula for unequal sample sizes
def required_sample_size2(alpha, beta, t0, sigma, treatment_share, effect_size_std=None):
    z_alpha = norm.ppf(1- alpha/2)
    z_beta = norm.ppf(beta)
    if effect_size_std is None: 
        n = ((z_alpha + z_beta)**2) / ((t0**2/sigma**2) * treatment_share * (1-treatment_share))
    else:
        n = ((z_alpha + z_beta)**2) / ((effect_size_std)**2 * treatment_share * (1-treatment_share))
    return n

min_n_v2 = required_sample_size2(0.05, 0.8, 0.05, std_exp_lwage, 0.5)
print(f'Minimum sample size for version 2: {min_n_v2}')

#1.c.
#to-do. now sure how to do this

#1.d.
##Repeat 1.b for the wage variable instead of the lwage variable
t0 = np.exp(5.94) - np.exp(5.9)
one_d_a = required_sample_size2(0.05, 0.8, t0, std_exp_wage, 0.5)
print(f'Minimum sample size using wage: {one_d_a}')
