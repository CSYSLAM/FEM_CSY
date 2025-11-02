# torchsisso_f1_recover_safe.py
import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from fractions import Fraction
from TorchSisso import SissoModel

# ----------------------
# 参数设置
# ----------------------
CSV = "f1_only_dataset.csv"
FEATURES = ["dotB1","dotB2","dotB3","B1B2","B1B3"]
EPS = 1e-12
MAX_DENOM = 50
N_RANDOM_RUNS = 5  # 多次随机搜索
N_EXPANSION = 2    # 安全 n_expansion，避免内存爆炸
K = 200            # 候选池
# ----------------------

# 读取数据
df = pd.read_csv(CSV)
features = [f for f in FEATURES if f in df.columns]
if len(features) == 0:
    raise SystemExit("❌ 没有找到任何特征列，请检查 FEATURES 与 CSV 列名是否一致。")

X_raw = df[features].values.astype(float)
y = df["f1"].values.astype(float)

# 目标变换
t = 1.0 / (y**2 + EPS)

# 标准化特征
X_mean = X_raw.mean(axis=0)
X_std = X_raw.std(axis=0)
X_std[X_std == 0] = 1.0
X_scaled = (X_raw - X_mean) / X_std

# 中心化目标
t_mean = t.mean()
t_centered = t - t_mean

# 构造 DataFrame
df_model = pd.concat([pd.Series(t_centered, name="t"), pd.DataFrame(X_scaled, columns=features)], axis=1).reset_index(drop=True)

# ----------------------
# 多次随机搜索
# ----------------------
best_r2 = -np.inf
best_sm_result = None
best_seed = None

for seed in range(N_RANDOM_RUNS):
    np.random.seed(seed)  # 设置全局随机性

    sm = SissoModel(
        data=df_model,
        operators=['+','-'],  # 线性组合
        n_expansion=N_EXPANSION,
        n_term=1,
        k=K,
        use_gpu=False
    )

    rmse, eq, r2, extra = sm.fit()
    print(f"Run {seed}: R² = {r2:.6f}, eq = {eq}")

    if r2 > best_r2:
        best_r2 = r2
        best_sm_result = (rmse, eq, r2, extra)
        best_seed = seed

rmse, eq, r2, extra = best_sm_result
print("\n✅ 最优随机种子:", best_seed)
print("🔥 SISSO 原始表达式（中心化目标）:", eq)

# ----------------------
# OLS 精修，恢复完整线性组合
# ----------------------
# 这里选择所有原始特征进行OLS，保证最终五个特征组合
X_sel = X_raw  # 原始尺度
coef, *_ = lstsq(X_sel, t, rcond=None)
coef = coef.flatten()
intercept = 0.0  # 理论上 intercept≈0

# 有理数近似 + 四舍五入整数
rats = [Fraction(float(c)).limit_denominator(MAX_DENOM) for c in coef]
ints = [int(round(float(r))) for r in rats]

# 输出结果
print("\n📊 OLS 精修系数 (原始尺度):")
for f, c, i in zip(FEATURES, coef, ints):
    print(f"  {f}: {c:.6f}, round -> {i}")

expr_linear = " + ".join(f"{i}*{f}" for i,f in zip(ints, FEATURES) if i != 0)
print("\n✨ 候选线性组合:")
print("t ≈", expr_linear)

print("\n📘 对应 f1 形式:")
print("f1 ≈ 1 / sqrt(" + expr_linear + ")")
