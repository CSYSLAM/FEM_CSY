import numpy as np
import pandas as pd
from TorchSisso import SissoModel

# -----------------------------
# 生成数据
# -----------------------------
np.random.seed(42)

# 二次多项式 y = 3*x^2 + 2*x + 1
X_poly = np.linspace(-5,5,100).reshape(-1,1)
y_poly = 3*X_poly**2 + 2*X_poly + 1 + 0.1*np.random.randn(*X_poly.shape)
df_poly = pd.DataFrame(np.hstack([y_poly, X_poly]), columns=["Target","F1"])

# 有理式 y = (2*x^2 + 3)/(x-1)
X_rat = np.linspace(1.1,5,100).reshape(-1,1)
y_rat = (2*X_rat**2 + 3)/(X_rat-1) + 0.1*np.random.randn(*X_rat.shape)
df_rat = pd.DataFrame(np.hstack([y_rat, X_rat]), columns=["Target","F1"])

# -----------------------------
# SISSO 参数
# -----------------------------
N_EXPANSION = 3  # 特征展开层数
K = 20           # SIS 阶段保留特征数量

# -----------------------------
# 模型训练
# -----------------------------
# 二次多项式
sm_poly = SissoModel(
    data=df_poly,
    operators=['+','-','*','pow(2)'],
    n_expansion=N_EXPANSION,
    n_term=1,
    k=K,
    use_gpu=False
)
rmse_poly, eq_poly, r2_poly, _ = sm_poly.fit()
print("二次多项式 RMSE:", rmse_poly)
print("二次多项式 方程:", eq_poly)
print("二次多项式 R2:", r2_poly)

# 有理式
sm_rat = SissoModel(
    data=df_rat,
    operators=['+','-','*','/','pow(2)'],
    n_expansion=N_EXPANSION,
    n_term=1,
    k=K,
    use_gpu=False
)
rmse_rat, eq_rat, r2_rat, _ = sm_rat.fit()
print("有理式 RMSE:", rmse_rat)
print("有理式 方程:", eq_rat)
print("有理式 R2:", r2_rat)
