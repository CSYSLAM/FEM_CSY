from fractions import Fraction

# 准备点（0.40..0.49）
xs_frac = [Fraction(40+i,100) for i in range(10)]
ys_frac = [
    Fraction(925, 184),
    Fraction(2656503, 563561),
    Fraction(2001924, 452813),
    Fraction(3236675, 780467),
    Fraction(188750, 48521),
    Fraction(824800, 226071),
    Fraction(2154587, 629864),
    Fraction(1758721, 548620),
    Fraction(454375, 151344),
    Fraction(1980718, 705051)
]

# 拟合函数 (整数系数形式)
def f_int_frac(x_frac):
    t = x_frac * x_frac
    num = Fraction(-44,1) + Fraction(90,1) * t
    den = Fraction(-44,1) * t + Fraction(45,1) * t * t
    return num, den, num / den

# 输出逐点结果
print(f"{'x':>5} {'original_fraction':>25} {'fitted_fraction':>25} {'diff (fraction)':>25} {'diff (float)':>15}")
for x, y in zip(xs_frac, ys_frac):
    num, den, val = f_int_frac(x)
    diff = val - y
    print(f"{float(x):5.2f} {str(y):>25} {str(val):>25} {str(diff):>25} {float(diff):15.3e}")

# 计算最大绝对误差（有理）
difs = [abs(f_int_frac(x)[2] - y) for x, y in zip(xs_frac, ys_frac)]
max_abs = max(difs)
print("\nMax absolute error (exact fraction):", max_abs, "≈", float(max_abs))
