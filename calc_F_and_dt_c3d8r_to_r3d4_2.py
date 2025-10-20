from fractions import Fraction
import numpy as np
## 这里时c3d8r单元与c3d4单元接触力与dt的关系拟合，第二段，六面体长度均为1
# ---------- data ----------
xs1 = np.array([0.20,0.25,0.30,0.35,0.40,0.45], dtype=float)
ys1 = np.array([Fraction(1225, 52), Fraction(2336, 161), Fraction(16600, 1737),
       Fraction(234400, 35917), Fraction(775, 172), Fraction(157600, 51597)], dtype=float)

# f1(x) = (-11 + 30 x^2) / (-11 x^2 + 15 x^4)
def f1(x):
    t = x*x
    num = -11 + 30*t
    den = -11*t + 15*(t*t)
    return num/den

print("Dataset1 verification (candidate f1):")
for xi, yi in zip(xs1, ys1):
    fx = f1(xi)
    print(f"x={xi:.2f}, original={float(yi):.12g}, f1={fx:.12g}, diff={fx-float(yi):.12g}")

