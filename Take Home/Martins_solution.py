# AdvTSA Takehome
# 1071919

import numpy as np
import statsmodels.tsa.api as tsa
from sklearn.metrics import mean_squared_error
from numpy.linalg import matrix_power as mp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import statsmodels.tsa.stattools as sts
import statsmodels.graphics.tsaplots as tsaplots

"""
1. In this problem you explore the finite sample properties of information criteria for lag selection
and its implication for forecasting accuracy using a small Monte Carlo simulation.
a) Repeat the steps 1b) to 1d) for T = 50, 100, 200.
"""

"""
b) Generate M = 1000 sets of time series data of length T + 4
"""


def generate_var_1(coefs, cov, num_steps=100):
    # Initialize the time series with zeros
    ts = np.zeros((num_steps, len(coefs)))

    # Iterate over time steps
    for t in range(1, num_steps):
        # Compute the current values of the time series
        curr_values = np.dot(ts[t - 1], coefs)

        # Generate normally distributed residuals with the specified mean and covariance
        residuals = np.random.multivariate_normal([0] * len(cov), cov)

        # Update the values of the time series
        ts[t] = curr_values + residuals
    return ts[50:] # Discard first 50 values


def generate_ts(T: int, alpha=0.5):
    M = 1000
    l = T + 50 + 4
    coefs = [[alpha, 0], [0.5, 0.5]]
    cov = [[1, 0.5], [0.5, 1]]
    ts_data = np.zeros((M, l-50, 2))
    for i in range(M):
        ts_data[i] = generate_var_1(coefs, cov, num_steps=l)
    return ts_data


ts_50 = generate_ts(50)
ts_100 = generate_ts(100)
ts_200 = generate_ts(200)

"""c) In each of the M replications, hold out the last 4 observations for forecasting and apply
the standard information criteria for lag selection (FPE, AIC, HQ, SC) with pmax = 8 in
a VAR with an intercept and store the selected lag length. Based on the selected model,
compute and store h-step ahead forecasts for h = 1, 2, 4. Using the results from all M
replications, report the relative frequency distribution of estimated VAR lag orders and
the normalized mean squared forecast errors,
where Σy,m(h) is the h-step ahead forecast error variance matrix in replication m based
on the true DGP parameters."""

"""d) In each of the M replications, also compute 95% prediction intervals1 for the predictions
in 1c) and check (for each horizon separately) whether the actual value is inside
the interval or not. Report the empirical interval coverage over the M replications for
each variable, VAR selection criteria and horizon h = 1, 2, 4."""

#  I combined c and d for simplicity reasons


def rel_freq_dis(ts_data):
    M = 1000
    # Initialize arrays to store the results
    lag_orders = {'fpe': np.zeros(M), 'aic': np.zeros(M), 'hq': np.zeros(M), 'sc': np.zeros(M)}
    mspe = {'fpe': np.zeros((M, 3)), 'aic': np.zeros((M, 3)), 'hq': np.zeros((M, 3)), 'sc': np.zeros((M, 3))}
    coverage_1 = {'fpe': np.zeros((M, 3)), 'aic': np.zeros((M, 3)), 'hq': np.zeros((M, 3)), 'sc': np.zeros((M, 3))}
    coverage_2 = {'fpe': np.zeros((M, 3)), 'aic': np.zeros((M, 3)), 'hq': np.zeros((M, 3)), 'sc': np.zeros((M, 3))}

    # Iterate over the M replications
    for m in range(M):
        # Hold out the last 4 observations for forecasting
        train = ts_data[m, :-4]
        test = ts_data[m, -4:]
        # Iterate over the possible lag orders
        scores = {'fpe': {}, 'aic': {}, 'hq': {}, 'sc': {}}
        for p in range(1, 9):
            # Fit a VAR model with the given lag order and an intercept
            model = tsa.VAR(train).fit(p, trend='n')

            # Compute the FPE score for the model
            scores['fpe'][p], scores['aic'][p], scores['hq'][p], scores['sc'][p] = model.fpe, model.aic, model.hqic, model.bic

        # Select the lag order that minimizes the FPE score
        for key in scores.keys():
            s = scores[key]
            p = min(s, key=s.get)
            lag_orders[key][m] = p

            # Fit a VAR model with the selected lag order and an intercept
            model = tsa.VAR(train).fit(p, trend='c')

            # Compute the h-step ahead forecasts
            forecast = model.forecast(train, steps=4)
            # Compute the MSPE for each h
            for index, h in enumerate([1,2,4]):
                y_true = test[h - 1].reshape(2, 1)
                y_pred = forecast[h - 1].reshape(2, 1)
                # Compute the MSE of the residuals
                # Extract the coefficient matrix for lag h-1
                coefs_old = model.params
                coefs = coefs_old.T
                k, kp = coefs.shape
                # B
                K = 2
                row0 = np.hstack((np.ones((1, 1)), np.zeros((1, K * p))))
                rowz = np.hstack((np.zeros((K * (p - 1), 1)), np.identity(K * (p - 1)), np.zeros((K * (p - 1), K))))
                B = np.vstack((row0, coefs, rowz))
                # J
                J = np.hstack((np.zeros((K, 1)), np.identity(K), np.zeros((K, K * (p - 1)))))
                sigma_u = model.sigma_u

                sigma_hat_yh = 0
                for i in range(h):  # formula at p. 64
                    PHIi = J @ mp(B, i) @ J.T
                    part_of_sum = PHIi @ sigma_u @ PHIi.T
                    sigma_hat_yh += part_of_sum
                mspe[key][m, index] = ((y_true - y_pred).T @ np.linalg.inv(sigma_hat_yh) @ (y_true - y_pred))[0, 0]

            for index, h in enumerate([1,2,4]):
                # Compute the lower and upper bounds of the 95% prediction interval
                mean = forecast[h-1]
                mse = mean_squared_error(test[:h], forecast[:h])
                se = np.sqrt(mse/(h+1))
                lower = mean - 1.96 * se
                upper = mean + 1.96 * se
                # Check if the actual value is inside the interval
                if lower[0] <= test[h-1][0] <= upper[0]:
                    coverage_1[key][m, index] = 1
                elif lower[1] <= test[h-1][1] <= upper[1]:
                    coverage_2[key][m, index] = 1

    # Compute the relative frequency distribution of the estimated VAR lag orders
    rel_freq, nmse, mean_coverage_1, mean_coverage_2 = {}, {}, {}, {}
    for key in lag_orders.keys():
        rel_freq[key] = np.bincount(lag_orders[key].astype(int)) / M

        # Compute the normalized mean squared forecast errors
        nmse[key] = np.mean(mspe[key], axis=0)

        # Compute the mean empirical interval coverage over the M replications for each horizon
        mean_coverage_1[key] = np.mean(coverage_1[key], axis=0)
        mean_coverage_2[key] = np.mean(coverage_2[key], axis=0)
    return rel_freq, nmse, mean_coverage_1, mean_coverage_2


# Convert the dictionary to a Pandas DataFrame
def plot_exercise1(rel_50, nmse_50, mean_coverage_1_50, mean_coverage_2_50, rel_100, nmse_100, mean_coverage_1_100, mean_coverage_2_100, rel_200, nmse_200, mean_coverage_1_200, mean_coverage_2_200, alpha):
    sns.set_style("whitegrid")
    sns.set(rc={"figure.figsize":(6, 4)})
    for t, d in [('50', rel_50), ('100',rel_100), ('200',rel_200)]:
        df = pd.DataFrame.from_dict(d, orient='index').reset_index()
        df.columns = ['criterion', '0', '1', '2', '3', '4', '5', '6', '7', '8']
        df = df.drop('0', axis= 1)
        # Reshape the DataFrame with the melt function
        df_melt = df.melt(id_vars=['criterion'], var_name='lag', value_name='value')
        sns.barplot(x='lag', y='value', hue='criterion', data=df_melt)
        plt.title(f"T={t} lag distribution alpha={alpha}")
        plt.legend(loc='upper right')
        plt.savefig(f'images/{t}-lag_distribution-{alpha}.png')
        plt.show()

    sns.set_style("whitegrid")
    sns.set(rc={"figure.figsize":(6, 4)})
    for t, d in [('50', nmse_50), ('100',nmse_100), ('200',nmse_200)]:
        df = pd.DataFrame.from_dict(d, orient='index').reset_index()
        df.columns = ['criterion', '1', '2', '4']
        # Reshape the DataFrame with the melt function
        df_melt = df.melt(id_vars=['criterion'], var_name='h-step', value_name='nmse')
        sns.barplot(x='h-step', y='nmse', hue='criterion', data=df_melt)
        plt.title(f"T={t} Normalized Mean Squared Forecast Error alpha={alpha}")
        plt.legend(loc='upper left')
        plt.savefig(f'images/{t}-NMSE-{alpha}.png')
        plt.show()

    sns.set_style("whitegrid")
    sns.set(rc={"figure.figsize":(10, 4)})
    for t, d1, d2 in [('50', mean_coverage_1_50, mean_coverage_2_50), ('100',mean_coverage_1_100, mean_coverage_2_100), ('200',mean_coverage_1_200, mean_coverage_2_200)]:
        fig, axes = plt.subplots(1, 2)
        df1 = pd.DataFrame.from_dict(d1, orient='index').reset_index()
        df1.columns = ['criterion', '1', '2', '4']
        df2 = pd.DataFrame.from_dict(d2, orient='index').reset_index()
        df2.columns = ['criterion', '1', '2', '4']
        df1_melt = df1.melt(id_vars=['criterion'], var_name='h-step', value_name='mean_coverage')
        df2_melt = df2.melt(id_vars=['criterion'], var_name='h-step', value_name='mean_coverage')
        sns.barplot(x='h-step', y='mean_coverage', hue='criterion', data=df1_melt, ax = axes[0])
        sns.barplot(x='h-step', y='mean_coverage', hue='criterion', data=df1_melt, ax = axes[1])
        axes[0].set_title(f"T={t} y1 mean coverage alpha={alpha}")
        axes[1].set_title(f"T={t} y2 mean coverage alpha={alpha}")
        plt.legend(loc='upper right')
        plt.savefig(f'images/{t}-mean_coverage-{alpha}.png')
        plt.show()


rel_50, nmse_50, mean_coverage_1_50, mean_coverage_2_50 = rel_freq_dis(ts_50)
rel_100, nmse_100, mean_coverage_1_100, mean_coverage_2_100 = rel_freq_dis(ts_100)
rel_200, nmse_200, mean_coverage_1_200, mean_coverage_2_200 = rel_freq_dis(ts_200)
for key in rel_50.keys():
    if len(rel_50[key]) < 9:
        rel_50[key] = np.concatenate((rel_50[key], np.zeros(9 - len(rel_50[key]))))
rel_50_df = pd.DataFrame(rel_50)

ts_50_095 = generate_ts(50, alpha = 0.95)
ts_100_095  = generate_ts(100, alpha = 0.95)
ts_200_095  = generate_ts(200, alpha = 0.95)

rel_50_095, nmse_50_095, mean_coverage_1_50_095, mean_coverage_2_50_095 = rel_freq_dis(ts_50_095)
rel_100_095, nmse_100_095, mean_coverage_1_100_095, mean_coverage_2_100_095 = rel_freq_dis(ts_100_095)
rel_200_095, nmse_200_095, mean_coverage_1_200_095, mean_coverage_2_200_095 = rel_freq_dis(ts_200_095)

plot_exercise1(rel_50, nmse_50, mean_coverage_1_50, mean_coverage_2_50, rel_100, nmse_100, mean_coverage_1_100, mean_coverage_2_100, rel_200, nmse_200, mean_coverage_1_200, mean_coverage_2_200, alpha = 0.5)
plot_exercise1(rel_50_095, nmse_50_095, mean_coverage_1_50_095, mean_coverage_2_50_095, rel_100_095, nmse_100_095, mean_coverage_1_100_095, mean_coverage_2_100_095, rel_200_095, nmse_200_095, mean_coverage_1_200_095, mean_coverage_2_200_095, alpha = 0.95)



"""Exercise 2 In this problem you study the dynamic effects of monetary policy and exchange rate shocks. usmacroex.csv contains quarterly, seasonally adjusted US data for the period 1960Q1-
2022Q3. This file includes data on a chained-weighted GDP price index (GDPCTPI), the unemployment rate (UNRATE), the federal funds rate (FEDFUNDS) and the US Dollar/UK
Pound exchange rate as (DEXUSUK). Observations on the exchange rate are available from 1971Q1 only. Only use data until 2019Q4 (unless specified differently below)."""
# a

data_all = pd.read_csv("usmacroex.csv", index_col=0, parse_dates=True)
data = data_all[:-11]  # Only use the data until 2019Q4
data['INFt'] = 400 * (np.log(data['GDPCTPI']).diff())

sns.set_style("whitegrid")
sns.set(rc={"figure.figsize":(15, 15)})
fig, axes = plt.subplots(3, 1, constrained_layout = True)
sns.lineplot(x='DATE', y='INFt', data=data, ax = axes[0])
axes[0].set_title("Annualized rate of inflation")

sns.lineplot(x='DATE', y="UNRATE", data=data, ax = axes[1])
axes[1].set_title("Unemployment rate")

sns.lineplot(x='DATE', y="FEDFUNDS", data=data, ax = axes[2])
axes[2].set_title("Federal funds rate")
plt.savefig(f'images/2a_lineplots.png')
plt.show()


# Conduct an ADF test
data_new = data[1:]
# Here constant only no trend
result_inft = adfuller(data_new['INFt'], maxlag=8, autolag='AIC', regression='c', regresults=True)
print(f"INFt - ADF statistic: {result_inft[0]}, p-value: {result_inft[1]}, 5% significance level: {result_inft[2]['5%']}")

result_unrate = adfuller(data_new['UNRATE'], maxlag=8, autolag='AIC', regression='c', regresults=True)
print(f"UNRATE - ADF statistic: {result_unrate[0]}, p-value: {result_unrate[1]}, 5% significance level: {result_unrate[2]['5%']}")

result_fedfunds = adfuller(data_new['FEDFUNDS'], maxlag=8, autolag='AIC', regression='c', regresults=True)
print(f"FEDFUNDS - ADF statistic: {result_fedfunds[0]}, p-value: {result_fedfunds[1]}, 5% significance level: {result_fedfunds[2]['5%']}")

# First Difference ADF Test
data_new['Delta_INFt'] = data_new['INFt'].diff()
data_new['Delta_UNRATE'] = data_new['UNRATE'].diff()
data_new['Delta_FEDFUNDS'] = data_new['FEDFUNDS'].diff()
data_new = data_new[1:] # First row NaN again

result_delta_inft = adfuller(data_new['Delta_INFt'], maxlag=8, autolag='AIC', regression='c', regresults=True)
print(f"Delta_INFt - ADF statistic: {result_delta_inft[0]}, p-value: {result_delta_inft[1]}, 5% significance level: {result_delta_inft[2]['5%']}")

result_delta_unrate = adfuller(data_new['Delta_UNRATE'], maxlag=8, autolag='AIC', regression='c', regresults=True)
print(f"Delta_UNRATE - ADF statistic: {result_delta_unrate[0]}, p-value: {result_delta_unrate[1]}, 5% significance level: {result_delta_unrate[2]['5%']}")

result_delta_fedfunds = adfuller(data_new['Delta_FEDFUNDS'], maxlag=8, autolag='AIC', regression='c', regresults=True)
print(f"Delta_FEDFUNDS - ADF statistic: {result_delta_fedfunds[0]}, p-value: {result_delta_fedfunds[1]}, 5% significance level: {result_delta_fedfunds[2]['5%']}")

# The Timeseries is non-stationary in the mean, but stationary after being differecened once

# b

y = data_new[['Delta_INFt', 'UNRATE', 'Delta_FEDFUNDS']]

# Select lag length using AIC
model_aic = tsa.VAR(y).fit(maxlags=12, ic='aic')
print(f'AIC: Selected lag length = {model_aic.k_ar}')

# Select lag length using SC
model_sc = tsa.VAR(y).fit(maxlags=12, ic='bic')
print(f'SC: Selected lag length = {model_sc.k_ar}')
# Test for residual autocorrelations
print(f'AIC: Portmanteau test for autocorrelation: {model_aic.test_whiteness().summary()}')
print(f'SC: Portmanteau test for autocorrelation: {model_sc.test_whiteness().summary()}')

# Test for normality of residuals
print(f'AIC: Normality test: p-value = {model_aic.test_normality().summary()}')
print(f'SC: Normality test: p-value = {model_sc.test_normality().summary()}')
plt.rcParams['figure.figsize'] = [10,5]
for r in model_aic.resid.columns:
    fig, axes = plt.subplots(2,2, constrained_layout = True)
    aic_residuals = model_aic.resid[r]
    sc_residuals = model_sc.resid[r]
    tsaplots.plot_acf(aic_residuals, ax=axes[0,0])
    tsaplots.plot_acf(sc_residuals, ax=axes[0,1])
    axes[0,0].set_title(f'AIC Autocorrelation {r}')
    axes[0,1].set_title(f'SC Autocorrelation {r}')
    axes[1,0].plot(model_aic.resid.index, aic_residuals)
    axes[1,1].plot(model_sc.resid.index, sc_residuals)
    plt.savefig(f'images/2b_residuals_{r}')
    plt.show()


"""
Use your preferred VAR from 2b) and check whether the federal funds rate Grangercauses
the remaining variables. Also test whether the inflation rate Granger-causes the
other variables in the system. Use a significance level of 1%. Interpret your results.
"""
# c 
# I'm going to go ahead with the AIC model
print(model_aic.test_causality(["Delta_INFt", "UNRATE"], "Delta_FEDFUNDS", kind='f', signif=0.01).summary())
print(model_aic.test_causality(["Delta_FEDFUNDS", "UNRATE"], "Delta_INFt", kind='f', signif=0.01).summary())


"""
Provide the estimated residual correlation matrix ˆRu of your preferred model. Given
the results, explain why using forecast error impulse responses in this system may be
misleading.
"""
# d
resid_corr = model_aic.resid_corr
print(resid_corr)



# e

irfs = model_aic.irf(24)
irfs.plot(orth=True,  signif=0.1)
plt.savefig(f'images/2e_shock1.png')
plt.show()


"""
Based on your VAR model, perform a forecast error variance decomposition of the
inflation, unemployment and federal funds rate. Comment on the relative importance
of the monetary policy shock.
"""
#f

fevd = model_aic.fevd(24)
print(fevd.summary())



"""
Repeat your analysis from 2e) using all variables in levels, i.e. redo your analysis with
yt = (INFt,UNRATEt,FEDFUNDSt)′. Compare the results from the impulse response
analysis to those in 2e).
"""
# h

model_levels = tsa.VAR(data[1:][['INFt', 'UNRATE', 'FEDFUNDS']])
results_levels = model_levels.fit(maxlags=12, ic='aic')

# Compute and plot impulse responses
irfs_levels = results_levels.irf(24)
irfs_levels.plot(orth= True, signif = 0.1)
plt.savefig(f'images/2h_shock1.png')
plt.show()

"""
Now add the exchange rate as a fourth variables to the VAR, i.e. use
yt = (ΔINFt,UNRATEt,ΔFEDFUNDSt,DEXUSUKt)′.
Compute orthogonalized IRFs to innovations in the federal funds rate and the exchange
rate from a VAR(m), where m is selected by AIC. Report responses for INFt,
UNRATEt, FEDFUNDSt, and DEXUSUKt.
"""

# i
# Add Exchange Rate as a variable:
data_full = data_new[42:] # DEXUSUK only starts in 1971
data_full['DEXUSUK'] = data_full['DEXUSUK'].astype(float)
model_all = tsa.VAR(data_full[['Delta_INFt', 'UNRATE', 'Delta_FEDFUNDS', 'DEXUSUK']])
result_all = model_all.fit(maxlags=12, ic='aic')

irfs_all = result_all.irf(24)
irfs_all.plot(orth=True, signif = 0.1)
plt.savefig(f'images/2i_shock_all.png')
plt.show()

# ii
fevd_all = result_all.fevd(24)
print(fevd_all.summary())

# iii
data_cut = data_full[:-48] # Data only until 2007Q4
model_cut = tsa.VAR(data_cut[['Delta_INFt', 'UNRATE', 'Delta_FEDFUNDS', 'DEXUSUK']])
result_cut = model_cut.fit(maxlags=12, ic='aic')
irfs_cut = result_cut.irf(24)
irfs_cut.plot(orth=True, signif = 0.1)
plt.savefig(f'images/2i_shock_cut.png')
plt.show()
