# generate_c3d8_Q0_dataset.py

import numpy as np
import pandas as pd
from scipy.special import roots_legendre

def N_all(r, s, t):
    return np.array([
        (1-r)*(1-s)*(1-t),
        r*(1-s)*(1-t),
        r*s*(1-t),
        (1-r)*s*(1-t),
        (1-r)*(1-s)*t,
        r*(1-s)*t,
        r*s*t,
        (1-r)*s*t
    ], dtype=float)

def dN_drst(r, s, t):
    dN_dr = np.array([
        -(1-s)*(1-t),
        (1-s)*(1-t),
        s*(1-t),
        -s*(1-t),
        -(1-s)*t,
        (1-s)*t,
        s*t,
        -s*t
    ], dtype=float)
    dN_ds = np.array([
        -(1-r)*(1-t),
        -r*(1-t),
        r*(1-t),
        (1-r)*(1-t),
        -(1-r)*t,
        -r*t,
        r*t,
        (1-r)*t
    ], dtype=float)
    dN_dt = np.array([
        -(1-r)*(1-s),
        -r*(1-s),
        -r*s,
        -(1-r)*s,
        (1-r)*(1-s),
        r*(1-s),
        r*s,
        (1-r)*s
    ], dtype=float)
    return dN_dr, dN_ds, dN_dt

def jacobian_at(r, s, t, nodes):
    dN_dr, dN_ds, dN_dt = dN_drst(r, s, t)
    J = np.zeros((3,3), dtype=float)
    for i in range(8):
        x, y, z = nodes[i]
        J[0,0] += dN_dr[i]*x
        J[0,1] += dN_ds[i]*x
        J[0,2] += dN_dt[i]*x
        J[1,0] += dN_dr[i]*y
        J[1,1] += dN_ds[i]*y
        J[1,2] += dN_dt[i]*y
        J[2,0] += dN_dr[i]*z
        J[2,1] += dN_ds[i]*z
        J[2,2] += dN_dt[i]*z
    return J

def gauss_points_weights(n):
    x, w = roots_legendre(n)
    return 0.5*(x + 1.0), 0.5*w

def compute_Q(nodes, quad_n=5):
    pts, ws = gauss_points_weights(quad_n)
    Q = np.zeros(8, dtype=float)
    for r in pts:
        for s in pts:
            for t in pts:
                wgt = ws[np.where(pts == r)[0][0]] * ws[np.where(pts == s)[0][0]] * ws[np.where(pts == t)[0][0]]
                N = N_all(r, s, t)
                J = jacobian_at(r, s, t, nodes)
                detJ = np.linalg.det(J)
                Q += wgt * N * detJ
    return Q * 8.0

# 生成数据
a_values = np.linspace(0.5, 1.4, 11)  # 11 points as in your example
data = []

for a in a_values:
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [a,   0.0, 1.0],
        [a,   a,   1.0],
        [0.0, a,   1.0]
    ], dtype=float)
    
    Q = compute_Q(nodes, quad_n=5)
    Veff = Q[0]  # Q_values[0]
    data.append({"a": a, "Veff": Veff})

df = pd.DataFrame(data)
df.to_csv("c3d8_Q0_dataset.csv", index=False)
print("✅ Saved c3d8_Q0_dataset.csv")
print(df.round(12))