# pysr_fit_from_Bcomponents.py
from pysr import PySRRegressor
import pandas as pd
import numpy as np
import sympy as sp

df = pd.read_csv('c3d4_Bcomponents_dataset.csv')

# Feature list: 12 component columns
components = []
for i in range(2,5):
    components += [f'B{i}_x', f'B{i}_y', f'B{i}_z']
# Optionally include norms and pairwise dots as features too (comment/uncomment as needed)
norms = [f'B{i}_norm' for i in range(1,5)]
pairdots = ['dotB1B2','dotB1B3','dotB1B4','dotB2B3','dotB2B4','dotB3B4']

# Choose final feature set to feed PySR (here we use the 12 raw components)
features = components  # or components + norms + pairdots

X = df[features].values
y = df['f1'].values

model = PySRRegressor(
    niterations=400,
    population_size=200,
    binary_operators=['+', '-', '*', '/'],
    unary_operators=["sqrt", "inv(x) = 1/x"],
    extra_sympy_mappings={"inv": lambda x: 1/x},
    maxsize=60,
    parsimony=1e-6,
    ncyclesperiteration=10,
    verbosity=1,
    procs=8,
    random_state=42
)

model.fit(X, y, variable_names=features)

print(model.equations_.head(20))
best = model.get_best()
print("Best raw equation:", best.equation)

# Sympy parse for postprocessing
sym_vars = sp.symbols(' '.join(features))
locals_map = {str(v): v for v in sym_vars}
locals_map.update({"inv": lambda x: 1/x})
sympy_expr = sp.sympify(best.equation, locals=locals_map)
sympy_expr = sp.simplify(sp.expand(sympy_expr))
print("Sympy f1:", sympy_expr)
