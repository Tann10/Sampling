# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 20:45:19 2020

"""
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
plt.style.use('default')

# function to generate a uniform stratified sample
def stratified_uniform_sample(m_per_bin ,n_bins):
    
    n = n_bins
    m = m_per_bin
    samples =[]
    bounds = [num/n for num in range(0,n+1)]

    for i in range(0, n):
        LB = bounds[i]
        UB = bounds[i+1]
        bin_sample = np.random.uniform(LB, UB, m)

        samples[(i*m+1):((i+1)*m)] = bin_sample

    return bounds[1:], samples

# Function for importance estimate of the population mean
def mean_estimator(importance_sample):
    est_values = importance_sample['sample'] * importance_sample['weights']
    estimate = np.mean(est_values)
    return estimate


# Function for importance estimate of the population standard deviation
def sd_estimator(importance_sample):
    n = importance_sample.shape[0]
    mean_est = mean_estimator(importance_sample)
    est_values = (importance_sample['sample'] - 
                  mean_est)**2 * importance_sample['weights']
    estimate = np.mean(est_values)*n/(n-1)
    estimate = estimate**0.5
    return estimate
    
# Function for importance estimate of the population Pr(X<bound)
def left_tail_estimator(importance_sample,bound):
    est_values = importance_sample['sample'] <bound
    est_values = est_values*importance_sample['weights']
    estimate = np.mean(est_values)
    return estimate


# Function for stratified importance sampling
def stratified_importance_sample(appro_mu, appro_sigma, 
                                 target_dist, m_per_bin, n_bins):
    
     importance_dist = norm(appro_mu, appro_sigma)
     strat_u_bounds, strat_u_samples = stratified_uniform_sample(m_per_bin, n_bins)
     
     # calculate inverse of CDF (percentage point function) of stratified samples
     importance_samples = importance_dist.ppf(strat_u_samples)
     
     importance_weights = target_dist.pdf(
             importance_samples)/importance_dist.pdf(importance_samples)
     
     sample_df = pd.DataFrame({'sample':importance_samples, 
                               'weights': importance_weights})
     return sample_df
 
# Function for importance and target distribution plots    
def plot_density(target_mu, appro_mu, target_sigma, appro_sigma, ax):
    
    target_x = np.linspace(target_mu - 4*target_sigma, 
                           target_mu + 4*target_sigma, 100)
    importance_x = np.linspace(appro_mu - 4*appro_sigma, 
                               appro_mu + 4*appro_sigma, 100)    
        
    ax.plot(target_x, norm.pdf(target_x, target_mu, target_sigma), 
            label = "target distribution", linewidth=1.5)
    
    ax.plot(importance_x, norm.pdf(importance_x, appro_mu, appro_sigma), 
            label = "importance distribution", linewidth=1.5)
    
    ax.set_ylabel('Density')
    ax.set_xlabel('X')

    ax.legend(loc=2)

# Function for CDF plots
def plot_CDF(samples, target_mu, appro_mu, target_sigma, appro_sigma, ax):
    
    # plot the cumulative histogram
    n, bins, patches = ax.hist(samples, density=True, histtype='step',
                               cumulative=True, label='Empirical', 
                               color='darkgreen')

    # Add a line showing the importance distribution.
    y = ((1 / (np.sqrt(2 * np.pi) * appro_sigma)) *
         np.exp(-0.5 * (1 / appro_sigma * (bins - appro_mu))**2))
    y = y.cumsum()
    y /= y[-1]

    ax.plot(bins, y, 'k--', linewidth=1.5, label='Theoretical Importance')

    # Add a line showing the target distribution.
    y = ((1 / (np.sqrt(2 * np.pi) * target_sigma)) *
         np.exp(-0.5 * (1 / target_sigma * (bins - target_mu))**2))
    y = y.cumsum()
    y /= y[-1]

    ax.plot(bins, y, 'r--', linewidth=1.5, label='Theoretical Target')
    ax.set_ylabel('Probability')
    ax.set_xlabel('X')
    ax.set_title('CDF')

    ax.legend()
    
    
if __name__ == "__main__":  
    
    n_bins =20
    m_per_bin = 100
    
    # original distribution
    target_mu = 0
    target_sigma = 1
    target_dist = norm(target_mu, target_sigma)
    
    # importance distribution
    appro_mu = -2
    appro_sigma = 1
    
    target_tail_prob = target_dist.cdf(appro_mu)

    imp_sample_df = stratified_importance_sample(appro_mu, appro_sigma, 
                                                 target_dist, m_per_bin, n_bins)
    
    # Evaluating Importance Estimates original Distribution
    mean_imp_estimate = mean_estimator(imp_sample_df)
    sd_imp_estimate = sd_estimator(imp_sample_df)
    p_imp_estimate = left_tail_estimator(imp_sample_df, appro_mu)
    
    # Evaluating Estimates Importance distribution
    mu_est = np.mean(imp_sample_df['sample'])
    sd_est = np.std(imp_sample_df['sample'])
    p_est = np.mean(imp_sample_df['sample']<appro_mu)
    
    # plot graph
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=[10,18])
    # adding title
    plt.figtext(0.05,0.92,'Stratified Gaussian Importance Sample of size: {},'\
              'n_bins = {}, # Samples per bin: {}'.format(n_bins*m_per_bin, 
              n_bins, m_per_bin), size=15)
    
    fig.suptitle(r'$\hat\mu_{} = {}, \hat\sigma_{} = {}, \hatp_{} ='\
                 '{}, \hat\mu_{} = {}, \hat\sigma_{} = {}, \hatp_{} ={}$'\
                 .format('q', round(mu_est, 4), 
                         'q', round(sd_est, 4), 
                         'q', round(p_est, 4),
                         'p', round(mean_imp_estimate, 4),
                         'p', round(sd_imp_estimate, 4),
                         'p', round(p_imp_estimate, 5)), y=0.91, size=14)

    # plot stratified samples and grid lines
    importance_dist = norm(appro_mu, appro_sigma)
    strat_u_bounds, strat_u_samples = stratified_uniform_sample(m_per_bin,
                                                                n_bins)
    boundary_list = importance_dist.ppf(strat_u_bounds)

    for idx in range(0, len(boundary_list)):
        ax[0].axvline(boundary_list[idx], 0, 1, linestyle = '--', 
          color= 'lightgray', linewidth=1.5)
    
    ax[0].scatter(imp_sample_df['sample'], 
      [0 for num in range(0,imp_sample_df.shape[0])], color='blue')
    
    #    plt.tight_layout()
    plot_density(target_mu, appro_mu, target_sigma, appro_sigma, ax[0])
    
    ax[0].axhline(0,color='darkgray')
    ax[0].axvline(appro_mu,color='darkgray')
    ax[0].axvline(target_mu,color='darkgray')

    # plot population parameters
    ax[0].text(1, 0.39, r'Pr(X < {}) = {}'.format(appro_mu, 
      round(target_tail_prob, 5)), fontsize=15)
    
    ax[0].text(2, 0.35, r'$\mu_{}={}$'.format('p', 
      round(float(target_mu), 1)), fontsize=15)
    
    ax[0].text(2, 0.31, r'$\sigma_{}={}$'.format('p',
      round(float(target_sigma),1)), fontsize=15)

    ax[0].get_shared_x_axes().join(ax[0], ax[1])

    # plot importance weights
#    ax[1].set_xlim(-6,4)
    ax[1].axhline(0,color='darkgray')
    ax[1].axhline(1,color='darkgray')
    ax[1].axvline(target_mu,color='darkgray')
    
    
    for idx in range(0, len(boundary_list)):
        ax[1].axvline(boundary_list[idx], 0, 1, linestyle = '--', 
          color= 'lightgray', linewidth=1.5)
        
    ax[1].scatter(imp_sample_df['sample'], imp_sample_df['weights'], 
      color='blue')
    ax[1].set_ylabel('Importance Weights')
    ax[1].set_xlabel('X')



    # plot CDF 
    plot_CDF(imp_sample_df['sample'], target_mu, appro_mu, target_sigma, 
             appro_sigma, ax[2])
    plt.savefig('stratified_importance_sample_2.png')

    
        
    
    
        
        
    
    
    
