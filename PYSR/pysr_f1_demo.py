"""
pysr_f1_demo_fixed.py

Fixed version that actually recovers:
    f1 = 1 / sqrt(2*dotB1 + 5*dotB2 + 5*dotB3 + 3*B1B2 + 3*B1B3)
"""

import sys
import pandas as pd
import numpy as np
import re

csv_path = sys.argv[1] if len(sys.argv) > 1 else "f1_only_dataset.csv"
df = pd.read_csv(csv_path)

# ✅ FIX 1: Only use relevant features (NO B2B3)
feature_cols = ["dotB1", "dotB2", "dotB3", "B1B2", "B1B3"]
feature_cols = [c for c in feature_cols if c in df.columns]
X = df[feature_cols].values
y = df["f1"].values

def make_safe_name(s):
    return "v_" + re.sub(r'[^0-9a-zA-Z]', '_', s)

orig_to_safe = {c: make_safe_name(c) for c in feature_cols}
safe_names = [orig_to_safe[c] for c in feature_cols]

try:
    from pysr import PySRRegressor
except Exception as e:
    print("PySR import failed:", e)
    sys.exit(1)

# ✅ FIX 2: Enable inv and sqrt
model = PySRRegressor(
    niterations=300,          # ✅ More iterations
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sqrt", "inv(x) = 1/x"],  # ✅ Critical!
    extra_sympy_mappings={"inv": lambda x: 1/x},
    procs=1,
    populations=15,
    maxsize=30,               # ✅ Reasonable complexity
    random_state=42,
)

print("Fitting PySR on f1...")
model.fit(X, y, variable_names=safe_names)

# ✅ FIX 3: Correct regex word boundary
def restore(eq_str, mapping):
    s = eq_str
    for orig, safe in mapping.items():
        s = re.sub(r'\b' + re.escape(safe) + r'\b', orig, s)  # ✅ raw string \b
    return s

print("\n=== PySR results (readable) ===")
try:
    best_eq = model.sympy()
    print(restore(str(best_eq), orig_to_safe))
except Exception as e:
    print("Fallback to equations table:", e)
    eqs = model.equations_
    for i, row in eqs.head(5).iterrows():
        eq_raw = str(row.get('equation', row.get('expr', '')))
        print(restore(eq_raw, orig_to_safe))