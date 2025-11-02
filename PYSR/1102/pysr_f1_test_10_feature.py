# pysr_fit_f1_all_features.py
from pysr import PySRRegressor
import pandas as pd
import numpy as np
import sympy as sp

# 读数据集
df = pd.read_csv('c3d4_f1_dataset.csv')

# 所有特征（单项 + 交叉项）
features = ['dotB1', 'dotB2', 'dotB3', 'dotB4',
            'dotB1B2', 'dotB1B3', 'dotB1B4',
            'dotB2B3', 'dotB2B4', 'dotB3B4']

X = df[features].values
y = df['f1'].values

# ==================== PySR 配置增强版 ====================
model = PySRRegressor(
    niterations=500,            # 增大迭代次数，提高搜索机会
    population_size=200,         # 增大种群，更充分搜索
    binary_operators=['+', '-', '*', '/'],   # 保留基本运算
    unary_operators=["sqrt", "inv(x) = 1/x"],  # 支持 sqrt 和 1/x
    extra_sympy_mappings={"inv": lambda x: 1/x},
    maxsize=50,                  # 控制复杂度
    parsimony=1e-6,              # 防止过拟合，同时保留必要复杂度
    ncyclesperiteration=10,
    verbosity=1,
    procs=8,                    # 使用多核加速（你可以根据 CPU 调整）
    random_state=42,
)

# ==================== 拟合 ====================
model.fit(X, y, variable_names=features)

# ==================== 查看结果 ====================
print(model.equations_.head(20))
best = model.get_best()
print("Best raw equation:", best.equation)

# ==================== SymPy 解析 ====================
sym_vars = sp.symbols(' '.join(features))
locals_map = {str(v): v for v in sym_vars}
locals_map.update({"inv": lambda x: 1/x})  # 让 sympy 识别 inv
sympy_expr = sp.sympify(best.equation, locals=locals_map)
sympy_expr = sp.simplify(sp.expand(sympy_expr))
print("Sympy f1:", sympy_expr)

# ==================== 数值验证 RMSE ====================
from sympy import lambdify
f1_func = lambdify(tuple(sym_vars), sympy_expr, 'numpy')
X_vals = [df[f].values for f in features]
f1_pred = f1_func(*X_vals)
rmse = np.sqrt(np.mean((f1_pred - y)**2))
print("RMSE:", rmse)
