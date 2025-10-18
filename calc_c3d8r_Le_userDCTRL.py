import numpy as np
from scipy.special import roots_legendre

def compute_L2_for_C3D8R(coords):
    """
    Compute characteristic length squared L^2 for Abaqus C3D8R element.
    
    Formula:
        L^2 = (1/2) * V / (S_center * detJ_center * 8)
    
    where:
        V = exact volume (computed via 3x3x3 Gauss integration)
        S_center = sum ||∇_x N_i||^2 at (ξ,η,ζ) = (0,0,0)
        detJ_center = det(J) at center
        8 = volume of reference element [-1,1]^3
    
    Parameters:
        coords: (8, 3) array-like, node coordinates in Abaqus C3D8 order:
            1: (-1,-1,-1)  2: (1,-1,-1)  3: (1,1,-1)  4: (-1,1,-1)
            5: (-1,-1, 1)  6: (1,-1, 1)  7: (1,1, 1)  8: (-1,1, 1)
    
    Returns:
        L2: float, characteristic length squared
    """
    coords = np.array(coords, dtype=float)
    assert coords.shape == (8, 3), "coords must be (8, 3)"
    
    # --- 1. Compute S_center and detJ_center at (0,0,0) ---
    xi, eta, zeta = 0.0, 0.0, 0.0
    
    # Natural coordinates of nodes
    nat_coords = np.array([
        [-1, -1, -1],
        [ 1, -1, -1],
        [ 1,  1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
        [ 1, -1,  1],
        [ 1,  1,  1],
        [-1,  1,  1],
    ], dtype=float)
    
    # Shape function derivatives in natural coordinates at center
    dN_dxi = np.zeros(8)
    dN_deta = np.zeros(8)
    dN_dzeta = np.zeros(8)
    
    for i in range(8):
        xi_i, eta_i, zeta_i = nat_coords[i]
        dN_dxi[i] = 0.125 * xi_i * (1 + eta_i * eta) * (1 + zeta_i * zeta)
        dN_deta[i] = 0.125 * eta_i * (1 + xi_i * xi) * (1 + zeta_i * zeta)
        dN_dzeta[i] = 0.125 * zeta_i * (1 + xi_i * xi) * (1 + eta_i * eta)
    
    # Build Jacobian J = dx/dξ
    J = np.zeros((3, 3))
    for i in range(8):
        x, y, z = coords[i]
        J[0, 0] += x * dN_dxi[i]   # dx/dξ
        J[1, 0] += y * dN_dxi[i]   # dy/dξ
        J[2, 0] += z * dN_dxi[i]   # dz/dξ
        
        J[0, 1] += x * dN_deta[i]  # dx/dη
        J[1, 1] += y * dN_deta[i]  # dy/dη
        J[2, 1] += z * dN_deta[i]  # dz/dη
        
        J[0, 2] += x * dN_dzeta[i] # dx/dζ
        J[1, 2] += y * dN_dzeta[i] # dy/dζ
        J[2, 2] += z * dN_dzeta[i] # dz/dζ
    
    detJ_center = np.linalg.det(J)
    if abs(detJ_center) < 1e-12:
        raise ValueError("Singular Jacobian at element center")
    
    # Compute S_center = sum ||∇_x N_i||^2
    J_inv = np.linalg.inv(J)
    S_center = 0.0
    for i in range(8):
        grad_nat = np.array([dN_dxi[i], dN_deta[i], dN_dzeta[i]])
        grad_phys = J_inv.T @ grad_nat
        S_center += np.dot(grad_phys, grad_phys)
    
    # --- 2. Compute exact volume V using 3x3x3 Gauss integration ---
    # Gauss points and weights for [-1, 1]
    gp1d, w1d = roots_legendre(3)  # 3-point Gauss-Legendre
    V = 0.0
    for i, xi_gp in enumerate(gp1d):
        for j, eta_gp in enumerate(gp1d):
            for k, zeta_gp in enumerate(gp1d):
                weight = w1d[i] * w1d[j] * w1d[k]
                
                # Compute Jacobian at this Gauss point
                J_gp = np.zeros((3, 3))
                for n in range(8):
                    xi_i, eta_i, zeta_i = nat_coords[n]
                    dN_dxi_n = 0.125 * xi_i * (1 + eta_i * eta_gp) * (1 + zeta_i * zeta_gp)
                    dN_deta_n = 0.125 * eta_i * (1 + xi_i * xi_gp) * (1 + zeta_i * zeta_gp)
                    dN_dzeta_n = 0.125 * zeta_i * (1 + xi_i * xi_gp) * (1 + eta_i * eta_gp)
                    
                    x, y, z = coords[n]
                    J_gp[0, 0] += x * dN_dxi_n
                    J_gp[1, 0] += y * dN_dxi_n
                    J_gp[2, 0] += z * dN_dxi_n
                    
                    J_gp[0, 1] += x * dN_deta_n
                    J_gp[1, 1] += y * dN_deta_n
                    J_gp[2, 1] += z * dN_deta_n
                    
                    J_gp[0, 2] += x * dN_dzeta_n
                    J_gp[1, 2] += y * dN_dzeta_n
                    J_gp[2, 2] += z * dN_dzeta_n
                
                detJ_gp = np.linalg.det(J_gp)
                V += weight * detJ_gp
    
    # --- 3. Compute L^2 ---
    E = S_center * detJ_center * 8.0  # 8 = volume of reference element
    L2 = 0.5 * V / E
    
    return L2


# -----------------------------
# 验证：梯形台
# -----------------------------
def test_special_case():
    from fractions import Fraction
    print("=== 验证梯形台单元 ===")
    
    a_vals = [0.1 + 0.1*i for i in range(15)]
    abaqus_fractions = [
        Fraction(148, 1083), Fraction(31, 201), Fraction(556, 3201), Fraction(52, 267), Fraction(28, 129),
        Fraction(49, 204), Fraction(292, 1107), Fraction(244, 849), Fraction(1084, 3489), Fraction(1, 3),
        Fraction(1324, 3729), Fraction(364, 969), Fraction(532, 1347), Fraction(109, 264), Fraction(76, 177)
    ]
    
    print(f"{'a':>4} {'Abaqus':>12} {'Computed':>12} {'Diff':>12}")
    print("-" * 45)
    
    for i, a in enumerate(a_vals):
        coords = [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [a, 0, 1],
            [a, a, 1],
            [0, a, 1],
        ]
        L2_computed = compute_L2_for_C3D8R(coords)
        L2_abaqus = float(abaqus_fractions[i])
        diff = abs(L2_computed - L2_abaqus)
        print(f"{a:4.1f} {L2_abaqus:20.16f} {L2_computed:20.16f} {diff:15.2e}")


# -----------------------------
# 测试：任意六面体
# -----------------------------
def test_arbitrary_hex():
    print("\n=== 任意六面体测试 ===")
    a = 1.5
    coords = [
        [0.0, 0.0, 0.0],
        [a, 0.0, 0.0],
        [a, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [a, 0.0, 1.0],
        [a, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
    L2 = compute_L2_for_C3D8R(coords)
    import math
    L= math.sqrt(L2)
    print(f"任意六面体 L = {L:.16f}")


if __name__ == "__main__":
    test_special_case()
    test_arbitrary_hex()