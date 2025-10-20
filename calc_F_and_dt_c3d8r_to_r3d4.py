from fractions import Fraction
## 这里时c3d8r单元与c3d4单元接触力与dt的关系拟合，第二段
# 原始分数数据（与你给出的一致）
fractions_list = [
    Fraction(490500, 73),
    Fraction(6625, 4),
    Fraction(151500, 211),
    Fraction(52875, 136),
    Fraction(3060, 13),
    Fraction(13875, 92),
    Fraction(91500, 931),
    Fraction(25875, 416),
    Fraction(14500, 417)
]

# 部分分式定义： f(x) = 6750*(1/x^2 + 1/(x^2 - 220))
def f_partial(x):
    t = Fraction(x*x, 1)
    return Fraction(6750, 1) * (Fraction(1, t) + Fraction(1, t - 220))

# 备用的原始有理式定义（等价）
def f_rational(x):
    num = Fraction(13500 * x * x - 1485000, 1)
    den = Fraction(x**4 - 220 * x * x, 1)
    return num / den

# 验证并打印差值（精确有理数）
print("x | original            | f_partial           | difference (f_partial - original)")
print("---|---------------------|---------------------|----------------------------------")
for i, x in enumerate(range(1, 10)):
    orig = fractions_list[i]
    fp = f_partial(x)
    diff = fp - orig
    print(f"{x:1d} | {str(orig):19s} | {str(fp):19s} | {str(diff)}")

# 最后检查 f_partial 与 f_rational 在这些点是否相等
equal_all = all(f_partial(x) == f_rational(x) for x in range(1,10))
print("\nPartial form equals rational form on x=1..9? ->", equal_all)
