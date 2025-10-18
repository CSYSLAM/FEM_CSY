import numpy as np
from scipy.special import roots_legendre

## F = C * 1/ETSTABLE^2 * rho * Qi
def N_all(r, s, t):
    """返回8个角节点的三线性形函数 N1..N8"""
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
    """返回8个节点形函数关于 r,s,t 的导数"""
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
    """计算给定节点坐标下的雅可比矩阵 3x3"""
    dN_dr, dN_ds, dN_dt = dN_drst(r, s, t)
    J = np.zeros((3,3), dtype=float)
    for i in range(8):
        xi, yi, zi = nodes[i]
        J[0,0] += dN_dr[i]*xi
        J[0,1] += dN_ds[i]*xi
        J[0,2] += dN_dt[i]*xi
        J[1,0] += dN_dr[i]*yi
        J[1,1] += dN_ds[i]*yi
        J[1,2] += dN_dt[i]*yi
        J[2,0] += dN_dr[i]*zi
        J[2,1] += dN_ds[i]*zi
        J[2,2] += dN_dt[i]*zi
    return J

def gauss_points_weights(n):
    """返回 [0,1] 上 Gauss-Legendre 积分点和权重"""
    x, w = roots_legendre(n)
    x_mapped = 0.5*(x + 1.0)
    w_mapped = 0.5*w
    return x_mapped, w_mapped

def compute_Q(nodes, quad_n=3):
    """
    输入: nodes (8x3 numpy array)
    输出: Q_i = 8 * ∫ N_i * detJ dr ds dt, i=1..8
    """
    pts, ws = gauss_points_weights(quad_n)
    Q = np.zeros(8, dtype=float)
    for i_r, r in enumerate(pts):
        for i_s, s in enumerate(pts):
            for i_t, t in enumerate(pts):
                wgt = ws[i_r] * ws[i_s] * ws[i_t]
                N = N_all(r,s,t)
                J = jacobian_at(r,s,t,nodes)
                detJ = np.linalg.det(J)
                Q += wgt * N * detJ
    Q *= 8.0  # 与之前定义一致
    return Q

if __name__ == "__main__":
    # 任意C3D8单元节点（8x3）
    nodes_example = np.array([
        [0.0,0.0,0.0],
        [1.0,0.0,0.0],
        [1.0,1.0,0.0],
        [0.0,1.0,0.0],
        [0.0,0.0,1.0],
        [0.7,0.0,1.0],
        [0.7,0.7,1.0],
        [0.0,0.7,1.0]
    ], dtype=float)

    Q_values = compute_Q(nodes_example, quad_n=5)  # 5-point Gauss per dim
    for i, Qi in enumerate(Q_values):
        print(f"Node {i+1}: Q = {Qi:.6f}")

