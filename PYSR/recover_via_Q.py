"""
recover_via_Q.py

Loads f1_only_dataset.csv (must be in same folder or pass path) and, using f1 as the target,
recovers the linear coefficients of Q by transforming f1 -> Q = 1/f1^2 and solving a least-squares
problem on physics-informed basis features:
    [dotB1, dotB2, dotB3, B1B2, B1B3, B2B3]

Usage:
    python recover_via_Q.py /path/to/f1_only_dataset.csv

Outputs recovered coefficients, fit errors, and prints symbolic f1 expression.
"""
import sys, numpy as np, pandas as pd, math
import sympy as sp

csv_path = sys.argv[1] if len(sys.argv)>1 else "f1_only_dataset.csv"
df = pd.read_csv(csv_path)

# Feature matrix
feature_cols = ["dotB1","dotB2","dotB3","B1B2","B1B3","B2B3"]
X = df[feature_cols].values
f1 = df["f1"].values
# transform target: Q = 1 / f1^2
Q = 1.0 / (f1**2)

# Solve linear least squares (no intercept)
coeffs, residuals, rank, s = np.linalg.lstsq(X, Q, rcond=None)
coeffs = coeffs.flatten()

print("Recovered coefficients (dotB1, dotB2, dotB3, B1B2, B1B3, B2B3):")
print(coeffs)

# Evaluate fit
Q_pred = X.dot(coeffs)
mae = np.mean(np.abs(Q - Q_pred))
mse = np.mean((Q - Q_pred)**2)
print(f"Q fit MAE={mae:.6e}, MSE={mse:.6e}, rank={rank}")

# Reconstruct f1 from predicted Q and compare
f1_pred = 1.0 / np.sqrt(np.maximum(Q_pred, 1e-24))
mae_f1 = np.mean(np.abs(f1 - f1_pred))
print(f"f1 fit MAE={mae_f1:.6e}")

# Symbolic expression
a,b,c,d,e,f = coeffs
dotB1_sym, dotB2_sym, dotB3_sym, B1B2_sym, B1B3_sym, B2B3_sym = sp.symbols('dotB1 dotB2 dotB3 B1B2 B1B3 B2B3')
Q_sym = a*dotB1_sym + b*dotB2_sym + c*dotB3_sym + d*B1B2_sym + e*B1B3_sym + f*B2B3_sym
f1_sym = 1/sp.sqrt(Q_sym)
print("\\nSymbolic recovered f1:")
sp.pprint(sp.simplify(f1_sym))

# Print first 6 comparisons
print("\\nFirst 6 samples (Q_true, Q_pred, f1_true, f1_pred):")
for i in range(min(6, len(df))):
    print(f"{df['Q'].iloc[i]:12.6e}  {Q_pred[i]:12.6e}  {df['f1'].iloc[i]:12.6e}  {f1_pred[i]:12.6e}")
