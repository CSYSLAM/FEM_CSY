import numpy as np
from itertools import product

def calcdNdxBar(xMat, ccc, vol):
    """计算 dNdxBar"""
    nI, nJ, nK = ccc.shape
    
    # 检查张量维度
    if nI != nJ or nJ != nK:
        print(f"Check dimension of tensor C! current dim={ccc.shape}")
        return None
    
    dNdxBar = np.zeros((nI, 3))
    
    for II in range(nI):
        for JJ in range(nJ):
            for KK in range(nK):
                term = np.array([
                    xMat[JJ, 1] * xMat[KK, 2],
                    xMat[JJ, 2] * xMat[KK, 0],
                    xMat[JJ, 0] * xMat[KK, 1]
                ]) * ccc[II, JJ, KK]
                dNdxBar[II] += term
    
    dNdxBar = dNdxBar / vol
    return [dNdxBar]  # 返回列表以匹配原代码

def calcCIJK(hourglassBase):
    """计算 CIJK 张量"""
    Lambda = hourglassBase["Λ"]
    Gamma = hourglassBase["Γ"]
    
    # 初始化 CCC 张量
    CCC = np.zeros((8, 8, 8))
    
    # Levi-Civita 张量 (3D)
    def levi_civita(i, j, k):
        if i == j or j == k or i == k:
            return 0
        indices = [i, j, k]
        if sorted(indices) != [0, 1, 2]:  # 调整为0-based索引
            return 0
        sign = 1
        for idx in range(len(indices)):
            for jdx in range(idx + 1, len(indices)):
                if indices[idx] > indices[jdx]:
                    sign *= -1
        return sign
    
    # 计算 CCC
    for II, JJ, KK in product(range(8), range(8), range(8)):
        for i, j, k in product(range(3), range(3), range(3)):
            term1 = 3 * Lambda[i, II] * Lambda[j, JJ] * Lambda[k, KK]
            term2 = Lambda[i, II] * Gamma[k, JJ] * Gamma[j, KK]
            term3 = Gamma[k, II] * Lambda[j, JJ] * Gamma[i, KK]
            term4 = Gamma[j, II] * Gamma[i, JJ] * Lambda[k, KK]
            
            CCC[II, JJ, KK] += (1/192.0) * levi_civita(i, j, k) * (term1 + term2 + term3 + term4)
    
    return CCC

# Hourglass 基向量
HourglassBaseVectors = {
    3: {
        "Σ": np.array([1, 1, 1, 1, 1, 1, 1, 1]),
        "Λ": np.array([
            [-1, 1, 1, -1, -1, 1, 1, -1],
            [-1, -1, 1, 1, -1, -1, 1, 1],
            [-1, -1, -1, -1, 1, 1, 1, 1]
        ]),
        "Γ": np.array([
            [1, 1, -1, -1, -1, -1, 1, 1],
            [1, -1, -1, 1, -1, 1, 1, -1],
            [1, -1, 1, -1, 1, -1, 1, -1],
            [-1, 1, -1, 1, 1, -1, 1, -1]
        ])
    },
    2: {
        "Σ": np.array([1, 1, 1, 1]),
        "Λ": np.array([
            [-1, 1, 1, -1],
            [-1, -1, 1, 1]
        ]),
        "Γ": np.array([
            [1, -1, 1, -1]
        ])
    }
}

# 元素信息（需要根据实际需求填充）
elemInfo = {
    "C3D8R": {"geo": "HEX", "nNode": 8, "dim": 3, "nIP": 1, "geoOrder": 1},
    "C3D8": {"geo": "HEX", "nNode": 8, "dim": 3, "nIP": 1, "geoOrder": 1}
}

# 积分点坐标和权重（简化实现，实际需要根据具体元素类型定义）
iPCoords = {
    "C3D8R": np.array([[0, 0, 0]]),  # 单点积分
    "C3D8": np.array([[0, 0, 0]])
}

iPWeights = {
    "C3D8R": np.array([8.0]),  # 对于单点积分，权重为8
    "C3D8": np.array([8.0])
}

def dNdξ(geoName, geoOrder, II, ξj, *coords):
    """形状函数对自然坐标的导数（简化实现）"""
    # 这里是简化版本，实际实现需要根据具体的形状函数
    if geoName == "HEX" and geoOrder == 1:
        xi, eta, zeta = coords[0], coords[1], coords[2]
        
        # 线性六面体单元的形状函数导数
        N_derivatives = [
            # dN/dξ for each node
            lambda xi, eta, zeta: -0.125 * (1-eta) * (1-zeta),
            lambda xi, eta, zeta:  0.125 * (1-eta) * (1-zeta),
            lambda xi, eta, zeta:  0.125 * (1+eta) * (1-zeta),
            lambda xi, eta, zeta: -0.125 * (1+eta) * (1-zeta),
            lambda xi, eta, zeta: -0.125 * (1-eta) * (1+zeta),
            lambda xi, eta, zeta:  0.125 * (1-eta) * (1+zeta),
            lambda xi, eta, zeta:  0.125 * (1+eta) * (1+zeta),
            lambda xi, eta, zeta: -0.125 * (1+eta) * (1+zeta),
            
            # dN/dη for each node
            lambda xi, eta, zeta: -0.125 * (1-xi) * (1-zeta),
            lambda xi, eta, zeta: -0.125 * (1+xi) * (1-zeta),
            lambda xi, eta, zeta:  0.125 * (1+xi) * (1-zeta),
            lambda xi, eta, zeta:  0.125 * (1-xi) * (1-zeta),
            lambda xi, eta, zeta: -0.125 * (1-xi) * (1+zeta),
            lambda xi, eta, zeta: -0.125 * (1+xi) * (1+zeta),
            lambda xi, eta, zeta:  0.125 * (1+xi) * (1+zeta),
            lambda xi, eta, zeta:  0.125 * (1-xi) * (1+zeta),
            
            # dN/dζ for each node
            lambda xi, eta, zeta: -0.125 * (1-xi) * (1-eta),
            lambda xi, eta, zeta: -0.125 * (1+xi) * (1-eta),
            lambda xi, eta, zeta: -0.125 * (1+xi) * (1+eta),
            lambda xi, eta, zeta: -0.125 * (1-xi) * (1+eta),
            lambda xi, eta, zeta:  0.125 * (1-xi) * (1-eta),
            lambda xi, eta, zeta:  0.125 * (1+xi) * (1-eta),
            lambda xi, eta, zeta:  0.125 * (1+xi) * (1+eta),
            lambda xi, eta, zeta:  0.125 * (1-xi) * (1+eta)
        ]
        
        # II 是节点索引 (0-based)，ξj 是坐标方向索引 (0-based)
        func_index = II + ξj * 8
        return N_derivatives[func_index](xi, eta, zeta)
    
    return 0.0

def calcVol(xMat, elType):
    """计算体积"""
    # 获取元素信息
    geoName = elemInfo[elType]["geo"]
    nNode = elemInfo[elType]["nNode"]
    nDim = elemInfo[elType]["dim"]
    nIP = elemInfo[elType]["nIP"]
    geoOrder = elemInfo[elType]["geoOrder"]
    
    vol = 0.0
    
    # 对每个积分点计算
    for ip in range(nIP):
        # 计算形状函数对自然坐标的导数
        dNdxi = np.zeros((nNode, nDim))
        
        if nDim == 2:
            xi, eta = iPCoords[elType][ip]
            for II in range(nNode):
                for ξj in range(nDim):
                    dNdxi[II, ξj] = dNdξ(geoName, geoOrder, II, ξj, xi, eta)
        elif nDim == 3:
            xi, eta, zeta = iPCoords[elType][ip]
            for II in range(nNode):
                for ξj in range(nDim):
                    dNdxi[II, ξj] = dNdξ(geoName, geoOrder, II, ξj, xi, eta, zeta)
        
        # 计算雅可比矩阵
        jac = np.dot(xMat.T, dNdxi)
        
        # 累加体积
        vol += np.linalg.det(jac) * iPWeights[elType][ip]
    
    return vol

def calcLe(xMat, elType):
    """计算特征长度"""
    if elType == "C3D8R":
        # 计算 CIJK
        cijk = calcCIJK(HourglassBaseVectors[3])
        
        # 计算 dNdxBar
        dNdxBar_result = calcdNdxBar(xMat, cijk, 1.0)  # vol=1 作为临时值
        if dNdxBar_result is None:
            return None
        
        dNdxBar = dNdxBar_result[0]
        
        # 计算 BB
        BB = 0.0
        for row in dNdxBar:
            BB += np.sum(row**2)
        
        # 计算体积
        V = calcVol(xMat, "C3D8")
        
        # 计算特征长度
        le = V / np.sqrt(2 * BB)
        return le
    else:
        print("Not supported element type!")
        return None

# 测试用例
if __name__ == "__main__":
    print("=== 测试用例 ===")
    
    a = 0.5
    xMat = np.array([
        [0.0, 0.0, 0.0],  # 节点1
        [a, 0.0, 0.0],  # 节点2
        [a, 1.0, 0.0],  # 节点3
        [0.0, 1.0, 0.0],  # 节点4
        [0.0, 0.0, 1.0],  # 节点5
        [a, 0.0, 1.0],  # 节点6
        [a, 1.0, 1.0],  # 节点7
        [0.0, 1.0, 1.0]   # 节点8
    ])
    
    print("节点坐标:")
    print(xMat)
    
    # 测试 calcVol
    volume = calcVol(xMat, "C3D8R")
    print(f"\n体积: {volume}")
    
    # # 测试 calcCIJK
    # cijk = calcCIJK(HourglassBaseVectors[3])
    # print(f"\nCIJK 张量形状: {cijk.shape}")
    # print(f"CIJK 张量部分值:")
    # for i in range(3):
    #     print(f"CCC[{i},0,0] = {cijk[i,0,0]}")
    
    # # 测试 calcdNdxBar
    # dNdxBar_result = calcdNdxBar(xMat, cijk, volume)
    # if dNdxBar_result is not None:
    #     print(f"\ndNdxBar 结果形状: {dNdxBar_result[0].shape}")
    #     print("dNdxBar 部分值:")
    #     for i in range(3):
    #         print(f"dNdxBar[{i}] = {dNdxBar_result[0][i]}")
    
    # 测试 calcLe
    le = calcLe(xMat, "C3D8R")
    print(f"\n特征长度 le: {le}")
    