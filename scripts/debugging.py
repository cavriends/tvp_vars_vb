import sys
import pandas as pd
import numpy as np

# Own code
sys.path.append("../")
from utils.data_utils import generate_contemp_matrices, transformation, standardize
from utils.tvp_models import tvp_ar_contemp_decomposition

# Suppress scientific notation in numpy
np.set_printoptions(suppress=True)

# Set M and standardization

M = 7
standardization = False
transform = True

ds = pd.read_csv("../data/fred_qd.csv")
gdp = transformation(ds["GDPC1"].iloc[2:].to_numpy(), 5, transform, scale=1)[2:]
cpi = transformation(ds["CPIAUCSL"].iloc[2:].to_numpy(), 6, transform, scale=1)[2:]
fedfund = transformation(ds["FEDFUNDS"].iloc[2:].to_numpy(), 2, transform, scale=1)[2:]
compi = transformation(ds["PPIACO"].iloc[2:].to_numpy(), 6, transform, scale=1)[2:]
borrowings = transformation(ds["TOTRESNS"].iloc[2:].to_numpy(), 6, transform, scale=1)[2:]
sp500 = transformation(ds["S&P 500"].iloc[2:].to_numpy(), 5, transform, scale=1)[2:]
m2 = transformation(ds["M2REAL"].iloc[2:].to_numpy(), 5, transform, scale=1)[2:]

# Start due to transformation
nonlagged_T = gdp.shape[0]
p = 1
T = nonlagged_T - p

if M == 3:

    series = [gdp, cpi, fedfund]

elif M == 7:

    series = [gdp, cpi, fedfund, compi, borrowings, sp500, m2]

if standardization:
    series = standardize(series, train=243 - 25)

series_total = np.array(series)

y_matrix_contemp, X_matrix_contemp = generate_contemp_matrices(nonlagged_T, M, p, series_total)

p = 1
prior = "lasso_alternative"
train = T - 25
prior_parameters = {"a0_lasso":1.1e-3, "b0_lasso":1e-3}

model_tvp = tvp_ar_contemp_decomposition(T, M, p, train, X_matrix_contemp, y_matrix_contemp, prior, print_status=False, iterations=50)

msfe, alpl, coeff, sigma = model_tvp.result()
