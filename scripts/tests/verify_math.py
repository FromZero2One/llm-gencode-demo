"""
验证 TRANSFORMER_MATH_FOUNDATION.md 中的所有数学计算和公式
"""

import sys
import os
import numpy as np
import torch

# 添加父目录（scripts）到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logger import logging_context

def verify_matrix_multiplication():
    """验证练习 1.1：矩阵乘法"""
    print("=" * 60)
    print("验证练习 1.1：矩阵乘法")
    print("=" * 60)
    
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])
    
    B = np.array([[7, 8],
                  [9, 10],
                  [11, 12]])
    
    C = A @ B
    
    expected = np.array([[58, 64],
                         [139, 154]])
    
    print(f"A shape: {A.shape}")
    print(f"B shape: {B.shape}")
    print(f"C = A @ B:\n{C}")
    print(f"\n期望结果:\n{expected}")
    print(f"结果正确: {np.allclose(C, expected)} ✓\n")
    
    return np.allclose(C, expected)


def verify_dot_product():
    """验证练习 1.2：点积与相似度"""
    print("=" * 60)
    print("验证练习 1.2：点积与相似度")
    print("=" * 60)
    
    u = np.array([1, 2, 3])
    v = np.array([4, 5, 6])
    
    # 点积
    dot_product = np.dot(u, v)
    print(f"u · v = {dot_product} (期望: 32)")
    
    # 范数
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    print(f"||u|| = {norm_u:.4f} (期望: 3.74)")
    print(f"||v|| = {norm_v:.4f} (期望: 8.77)")
    
    # 余弦相似度
    cos_theta = dot_product / (norm_u * norm_v)
    print(f"cos(θ) = {cos_theta:.4f} (期望: 0.975)")
    
    # 判断角度
    angle_type = "锐角" if cos_theta > 0 else ("直角" if cos_theta == 0 else "钝角")
    print(f"夹角类型: {angle_type} (期望: 锐角)")
    
    correct = (dot_product == 32 and 
               abs(norm_u - 3.74) < 0.01 and 
               abs(norm_v - 8.77) < 0.01 and
               abs(cos_theta - 0.975) < 0.01)
    print(f"结果正确: {correct} ✓\n")
    
    return correct


def verify_softmax():
    """验证练习 2.4：Softmax 计算"""
    print("=" * 60)
    print("验证练习 2.4：Softmax 计算")
    print("=" * 60)
    
    x = np.array([1.0, 2.0, 3.0])
    
    # 手动计算
    exp_x = np.exp(x)
    sum_exp = np.sum(exp_x)
    softmax_manual = exp_x / sum_exp
    
    print(f"x = {x}")
    print(f"exp(x) = {exp_x}")
    print(f"sum(exp(x)) = {sum_exp:.4f}")
    print(f"softmax(x) = {softmax_manual}")
    
    # 使用 NumPy 验证
    softmax_numpy = np.exp(x) / np.sum(np.exp(x))
    print(f"\nNumPy 验证: {softmax_numpy}")
    
    # 验证和为 1
    sum_check = np.sum(softmax_manual)
    print(f"\nsoftmax 元素和: {sum_check:.6f} (应该等于 1.0)")
    
    # 计算导数
    s = softmax_manual
    ds_dx_11 = s[0] * (1 - s[0])  # ∂s₁/∂x₁
    ds_dx_12 = -s[0] * s[1]       # ∂s₁/∂x₂
    
    print(f"\n∂softmax₁/∂x₁ = {ds_dx_11:.4f} (期望: 0.082)")
    print(f"∂softmax₁/∂x₂ = {ds_dx_12:.4f} (期望: -0.022)")
    
    correct = (abs(sum_check - 1.0) < 1e-6 and
               abs(ds_dx_11 - 0.082) < 0.01 and
               abs(ds_dx_12 + 0.022) < 0.01)
    print(f"结果正确: {correct} ✓\n")
    
    return correct


def verify_softmax_gradient():
    """验证 Softmax 梯度公式"""
    print("=" * 60)
    print("验证 Softmax 梯度公式")
    print("=" * 60)
    
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    s = torch.softmax(x, dim=0)
    
    # 计算 Jacobian 矩阵
    jacobian = torch.zeros(3, 3)
    for i in range(3):
        for j in range(3):
            # 数值梯度
            eps = 1e-5
            x_plus = x.clone()
            x_plus[j] += eps
            s_plus = torch.softmax(x_plus, dim=0)
            
            x_minus = x.clone()
            x_minus[j] -= eps
            s_minus = torch.softmax(x_minus, dim=0)
            
            jacobian[i, j] = (s_plus[i] - s_minus[i]) / (2 * eps)
    
    print("数值计算的 Jacobian 矩阵:")
    print(jacobian)
    
    # 理论公式：J[i,j] = s_i * (δ_ij - s_j)
    s_val = s.detach()
    jacobian_theory = torch.zeros(3, 3)
    for i in range(3):
        for j in range(3):
            delta_ij = 1.0 if i == j else 0.0
            jacobian_theory[i, j] = s_val[i] * (delta_ij - s_val[j])
    
    print("\n理论公式计算的 Jacobian 矩阵:")
    print(jacobian_theory)
    
    print(f"\n两者差异: {torch.max(torch.abs(jacobian - jacobian_theory)):.2e}")
    
    correct = torch.allclose(jacobian, jacobian_theory, atol=1e-2)
    print(f"梯度公式正确: {correct} ✓\n")
    
    return correct


def verify_attention_computation():
    """验证练习 5.1：完整的注意力计算"""
    print("=" * 60)
    print("验证练习 5.1：完整的注意力计算")
    print("=" * 60)
    
    Q = np.array([[1.0, 0.0],
                  [0.0, 1.0]])
    
    K = np.array([[1.0, 0.0],
                  [0.0, 1.0]])
    
    V = np.array([[1.0, 2.0],
                  [3.0, 4.0]])
    
    d_k = 2
    
    # Step 1: scores = Q @ K^T
    scores = Q @ K.T
    print(f"Step 1: scores = Q @ K^T")
    print(scores)
    
    # Step 2: scaled_scores = scores / √d_k
    scaled_scores = scores / np.sqrt(d_k)
    print(f"\nStep 2: scaled_scores = scores / √{d_k}")
    print(scaled_scores)
    
    # Step 3: attention_weights = softmax(scaled_scores)
    # 对每行分别计算 softmax
    attention_weights = np.zeros_like(scaled_scores)
    for i in range(2):
        exp_row = np.exp(scaled_scores[i])
        attention_weights[i] = exp_row / np.sum(exp_row)
    
    print(f"\nStep 3: attention_weights = softmax(scaled_scores)")
    print(attention_weights)
    
    # Step 4: output = attention_weights @ V
    output = attention_weights @ V
    print(f"\nStep 4: output = attention_weights @ V")
    print(output)
    
    # 验证结果
    expected_output = np.array([[1.660, 2.660],
                                [2.340, 3.340]])
    
    print(f"\n期望输出:")
    print(expected_output)
    
    correct = np.allclose(output, expected_output, atol=0.01)
    print(f"结果正确: {correct} ✓\n")
    
    return correct


def verify_entropy():
    """验证熵的计算"""
    print("=" * 60)
    print("验证熵的计算（练习 3.4）")
    print("=" * 60)
    
    # 公平硬币
    p_fair = np.array([0.5, 0.5])
    H_fair = -np.sum(p_fair * np.log2(p_fair))
    print(f"公平硬币的熵: {H_fair:.4f} bits (期望: 1.0)")
    
    # 不公平硬币
    p_unfair = np.array([0.9, 0.1])
    H_unfair = -np.sum(p_unfair * np.log2(p_unfair))
    print(f"不公平硬币的熵: {H_unfair:.4f} bits (期望: 0.469)")
    
    # 三面骰子
    p_dice = np.array([1/3, 1/3, 1/3])
    H_dice = -np.sum(p_dice * np.log2(p_dice))
    print(f"三面骰子的熵: {H_dice:.4f} bits (期望: 1.585)")
    
    correct = (abs(H_fair - 1.0) < 0.01 and
               abs(H_unfair - 0.469) < 0.01 and
               abs(H_dice - 1.585) < 0.01)
    
    print(f"\n最大熵的是: {'三面骰子' if H_dice > max(H_fair, H_unfair) else '其他'}")
    print(f"结果正确: {correct} ✓\n")
    
    return correct


def verify_cross_entropy():
    """验证交叉熵计算"""
    print("=" * 60)
    print("验证交叉熵计算（练习 4.1）")
    print("=" * 60)
    
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.6, 0.3, 0.1])
    
    # 交叉熵
    H_pq = -np.sum(p * np.log2(q))
    print(f"H(p, q) = {H_pq:.4f} bits (期望: 1.195)")
    
    # KL 散度
    D_KL = np.sum(p * np.log2(p / q))
    print(f"D_KL(p || q) = {D_KL:.4f} bits (期望: 0.038)")
    
    # 验证关系：H(p,q) = H(p) + D_KL(p||q)
    H_p = -np.sum(p * np.log2(p))
    print(f"\nH(p) = {H_p:.4f} bits")
    print(f"H(p) + D_KL = {H_p + D_KL:.4f} bits")
    print(f"H(p, q) = {H_pq:.4f} bits")
    print(f"关系验证: {abs(H_pq - (H_p + D_KL)) < 1e-6}")
    
    correct = (abs(H_pq - 1.195) < 0.01 and
               abs(D_KL - 0.038) < 0.01)
    print(f"结果正确: {correct} ✓\n")
    
    return correct


def verify_positional_encoding():
    """验证位置编码计算"""
    print("=" * 60)
    print("验证位置编码计算（练习 5.3）")
    print("=" * 60)
    
    pos = 2
    d_model = 4
    
    PE = np.zeros(d_model)
    
    # i = 0
    PE[0] = np.sin(pos / (10000 ** (0 / d_model)))
    PE[1] = np.cos(pos / (10000 ** (0 / d_model)))
    
    # i = 1
    PE[2] = np.sin(pos / (10000 ** (2 / d_model)))
    PE[3] = np.cos(pos / (10000 ** (2 / d_model)))
    
    print(f"PE({pos}) = {PE}")
    print(f"PE(2, 0) = sin(2) = {PE[0]:.4f} (期望: 0.909)")
    print(f"PE(2, 1) = cos(2) = {PE[1]:.4f} (期望: -0.416)")
    print(f"PE(2, 2) = sin(0.02) = {PE[2]:.4f} (期望: 0.020)")
    print(f"PE(2, 3) = cos(0.02) = {PE[3]:.4f} (期望: 1.000)")
    
    expected = np.array([0.909, -0.416, 0.020, 1.000])
    correct = np.allclose(PE, expected, atol=0.01)
    print(f"结果正确: {correct} ✓\n")
    
    return correct


def verify_chain_rule():
    """验证链式法则"""
    print("=" * 60)
    print("验证链式法则（练习 2.2）")
    print("=" * 60)
    
    # f(x) = e^(sin(x²))
    # f'(x) = 2x · cos(x²) · e^(sin(x²))
    
    x = 1.5
    
    # 数值导数
    eps = 1e-7
    f_plus = np.exp(np.sin((x + eps)**2))
    f_minus = np.exp(np.sin((x - eps)**2))
    numerical_grad = (f_plus - f_minus) / (2 * eps)
    
    # 解析导数
    analytical_grad = 2 * x * np.cos(x**2) * np.exp(np.sin(x**2))
    
    print(f"x = {x}")
    print(f"数值导数: {numerical_grad:.6f}")
    print(f"解析导数: {analytical_grad:.6f}")
    print(f"差异: {abs(numerical_grad - analytical_grad):.2e}")
    
    correct = abs(numerical_grad - analytical_grad) < 1e-5
    print(f"链式法则正确: {correct} ✓\n")
    
    return correct


def verify_expectation_variance():
    """验证期望和方差计算"""
    print("=" * 60)
    print("验证期望和方差（练习 3.1）")
    print("=" * 60)
    
    # P(X=1) = 0.2, P(X=2) = 0.5, P(X=3) = 0.3
    values = np.array([1, 2, 3])
    probs = np.array([0.2, 0.5, 0.3])
    
    E_X = np.sum(values * probs)
    E_X2 = np.sum(values**2 * probs)
    Var_X = E_X2 - E_X**2
    
    print(f"E[X] = {E_X:.2f} (期望: 2.1)")
    print(f"E[X²] = {E_X2:.2f} (期望: 4.9)")
    print(f"Var(X) = {Var_X:.2f} (期望: 0.49)")
    
    correct = (abs(E_X - 2.1) < 0.01 and
               abs(E_X2 - 4.9) < 0.01 and
               abs(Var_X - 0.49) < 0.01)
    print(f"结果正确: {correct} ✓\n")
    
    return correct


def main():
    """运行所有验证"""
    print("\n" + "=" * 60)
    print("TRANSFORMER_MATH_FOUNDATION.md 验证程序")
    print("=" * 60 + "\n")
    
    results = {}
    
    # 线性代数验证
    results['矩阵乘法'] = verify_matrix_multiplication()
    results['点积与相似度'] = verify_dot_product()
    
    # 微积分验证
    results['Softmax计算'] = verify_softmax()
    results['Softmax梯度'] = verify_softmax_gradient()
    results['链式法则'] = verify_chain_rule()
    
    # 概率论验证
    results['熵的计算'] = verify_entropy()
    results['期望和方差'] = verify_expectation_variance()
    
    # 信息论验证
    results['交叉熵'] = verify_cross_entropy()
    
    # 综合应用验证
    results['注意力计算'] = verify_attention_computation()
    results['位置编码'] = verify_positional_encoding()
    
    # 总结
    print("=" * 60)
    print("验证总结")
    print("=" * 60)
    
    total = len(results)
    passed = sum(results.values())
    
    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name:15s}: {status}")
    
    print("\n" + "=" * 60)
    print(f"总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有验证通过！文档中的数学计算完全正确！")
    else:
        print(f"⚠️  有 {total - passed} 个测试失败，请检查文档。")
    
    print("=" * 60 + "\n")
    
    return passed == total


if __name__ == "__main__":
    # 使用日志上下文管理器
    with logging_context(__file__):
        success = main()
        exit(0 if success else 1)
