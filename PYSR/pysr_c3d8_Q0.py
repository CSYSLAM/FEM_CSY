# pysr_c3d8_Q0.py

import sys
import pandas as pd
import numpy as np
import re

csv_path = sys.argv[1] if len(sys.argv) > 1 else "c3d8_Q0_dataset.csv"
df = pd.read_csv(csv_path)

X = df[["a"]].values
y = df["Veff"].values

try:
    from pysr import PySRRegressor
except Exception as e:
    print("PySR not installed:", e)
    sys.exit(1)

model = PySRRegressor(
    niterations=200,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=[],  # 不需要 sqrt, inv 等
    maxsize=20,
    populations=10,
    random_state=42,
    verbosity=0,
)

print("Fitting Veff = f(a) ...")
model.fit(X, y, variable_names=["a"])

print("\n🔍 Discovered formula:")
expr = model.sympy()
print(expr)

# 验证是否等于 (a**2 + 2*a + 3)/6
a = np.linspace(0.5, 1.4, 11)
y_true = (a**2 + 2*a + 3) / 6
y_pred = model.predict(X)

print(f"\n📊 Max error vs (a²+2a+3)/6: {np.max(np.abs(y_pred - y_true)):.2e}")