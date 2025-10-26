# run_gplearn_f1.py
import sys, numpy as np, pandas as pd
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
import warnings
warnings.filterwarnings("ignore")

csv_path = sys.argv[1] if len(sys.argv)>1 else "ai_feynman_f1_dataset.csv"
df = pd.read_csv(csv_path)
feature_cols = ["dotB1","dotB2","dotB3","B1B2","B1B3","B2B3"]
X = df[feature_cols].values
y = df["f1"].values

# protected sqrt and division
def _protected_sqrt(x):
    return np.sqrt(np.abs(x))
def _protected_div(x1, x2):
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(np.abs(x2) > 1e-12, x1 / x2, 1.0)
    return out

psqrt = make_function(function=_protected_sqrt, name='psqrt', arity=1)
pdiv = make_function(function=_protected_div, name='pdiv', arity=2)

est = SymbolicRegressor(
    population_size=2000,     # increase population
    generations=80,           # enough generations
    tournament_size=20,
    stopping_criteria=1e-8,
    p_crossover=0.7,
    p_subtree_mutation=0.1,
    p_hoist_mutation=0.05,
    p_point_mutation=0.1,
    max_samples=0.9,
    parsimony_coefficient=0.001,
    verbose=1,
    function_set=('add','sub','mul',pdiv,psqrt)
)

print("Fitting GP (this can take time) ...")
est.fit(X, y)
print("Best program:", est._program)
print("Fitness (OOB):", est._program.raw_fitness_)

# If GP struggles to find 1/sqrt(Q), try transform target to Q and fit linear model:
from sklearn.linear_model import LinearRegression
Q = 1.0 / (y**2)
lr = LinearRegression(fit_intercept=False).fit(X, Q)
print("Linear regression coefficients (dotB1,dotB2,dotB3,B1B2,B1B3,B2B3):", lr.coef_)
