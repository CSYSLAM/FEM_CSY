import sympy as sp

# === 全局定义符号（关键修复！）===
x1, y1, z1 = sp.symbols('x1 y1 z1')
x2, y2, z2 = sp.symbols('x2 y2 z2')
x3, y3, z3 = sp.symbols('x3 y3 z3')
x4, y4, z4 = sp.symbols('x4 y4 z4')

def c3d4_symbolic_characteristic_length():
    # 节点坐标矩阵 (4x3)
    nodes = sp.Matrix([
        [x1, y1, z1],
        [x2, y2, z2],
        [x3, y3, z3],
        [x4, y4, z4]
    ])

    # 形函数对参考坐标 (ξ, η, ζ) 的导数
    dN_dxi = sp.Matrix([
        [-1, -1, -1],
        [ 1,  0,  0],
        [ 0,  1,  0],
        [ 0,  0,  1]
    ])

    # === 构造雅可比矩阵 J (3x3) ===
    J = sp.zeros(3, 3)
    for i in range(4):
        coord_vec = sp.Matrix([nodes[i, 0], nodes[i, 1], nodes[i, 2]])
        dN_vec = sp.Matrix([dN_dxi[i, 0], dN_dxi[i, 1], dN_dxi[i, 2]])
        J += coord_vec * dN_vec.T

    detJ = sp.simplify(J.det())
    invJ = sp.simplify(J.inv())

    # === 形函数梯度 B[i] = ∇N_i ===
    B = []
    for i in range(4):
        dN_i_ref = sp.Matrix([dN_dxi[i, 0], dN_dxi[i, 1], dN_dxi[i, 2]])
        Bi = invJ * dN_i_ref
        B.append(sp.simplify(Bi))

    # === 特征长度计算 ===
    def dot(v):
        return (v.T * v)[0]

    min_L_list = [
        (sp.Rational(47, 75)) / sp.sqrt(dot(B[0])),
        (sp.Rational(47, 75)) / sp.sqrt(dot(B[1])),
        (sp.Rational(47, 75)) / sp.sqrt(dot(B[2])),
        (sp.Rational(47, 75)) / sp.sqrt(dot(B[3]))
    ]
    min_L = sp.Min(*min_L_list)

    Q = (
        2 * dot(B[1])
        + 5 * (dot(B[2]) + dot(B[3]))
        + 3 * ((B[1].T * B[2])[0] + (B[1].T * B[3])[0])
    )
    f1 = 1 / sp.sqrt(Q)
    Le = sp.Max(min_L, f1)

    # === 打印符号结果 ===
    print("=== C3D4 单元符号推导结果 ===\n")
    print("雅可比矩阵 J =")
    sp.pprint(J)
    print("\nJacobian 行列式 det(J) =")
    sp.pprint(detJ)
    print("\nJ 的逆矩阵 inv(J) =")
    sp.pprint(invJ)

    print("\n每个节点形函数梯度 B[i] = ∇N_i :")
    for i, Bi in enumerate(B):
        print(f"\nB[{i+1}] =")
        sp.pprint(Bi)

    print("\nmin_L（符号表达式）=")
    sp.pprint(min_L)

    print("\nf1（符号表达式）=")
    sp.pprint(f1)

    print("\n最终特征长度 Le = Max(min_L, f1)")
    sp.pprint(Le)

    return {
        "J": J,
        "detJ": detJ,
        "invJ": invJ,
        "B": B,
        "min_L": min_L,
        "f1": f1,
        "Le": Le
    }


# ==== 主程序 ====
if __name__ == "__main__":
    result = c3d4_symbolic_characteristic_length()

    # 使用全局定义的符号进行代入
    subs_dict = {
        x1: 0, y1: 0, z1: 0,
        x2: 1, y2: 0, z2: 0,
        x3: 0, y3: 1, z3: 0,
        x4: 0, y4: 0, z4: 1
    }

    print("\n" + "="*60)
    print("=== 代入标准四面体坐标验证 ===")
    print("="*60)

    for i in range(4):
        Bi_val = result["B"][i].subs(subs_dict).evalf()
        print(f"B[{i+1}] = {Bi_val.T}")

    Le_val = result["Le"].subs(subs_dict).evalf()
    print(f"\n特征长度 Le = {Le_val}")

    detJ_val = result["detJ"].subs(subs_dict).evalf()
    print(f"det(J) = {detJ_val} (应为 1.0)")