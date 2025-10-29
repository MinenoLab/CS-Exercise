import numpy as np
import matplotlib.pyplot as plt

# --- 活性化関数 ---
def step_function(x):
    return np.where(x >= 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    

# --- 単純パーセプトロン計算 ---
def simple_perceptron(input_vector, weight_vector, bias):
    # 1. 積和演算（内積）: ΣWk * xk
    weighted_sum = np.dot(input_vector, weight_vector)
    
    # 2. バイアスを引く: ΣWk * xk - b
    net_input = weighted_sum - bias
    
    # 3. 活性化関数（ステップ関数）を適用: H(ΣWk * xk - b)
    # net_input >= 0 なら 1，それ以外は 0 (P.8, P.11の「H」列の動作を再現)
    output = step_function(net_input)
    
    return weighted_sum, net_input, output

# --- MLP計算 ---
def multi_layer_perceptron(X, W_H, B_H, W_OUT, B_OUT):
    H_net = np.dot(X, W_H.T) - B_H
    H_out = step_function(H_net)
    Y_net = np.dot(H_out, W_OUT) - B_OUT
    Y_out = step_function(Y_net)
    return Y_out, H_out, H_net, Y_net


# --- テストパターン定義 ---
def get_test_patterns():
    X_Vertical   = np.array([1,0,0, 1,0,0, 1,0,0])
    X_Horizontal = np.array([0,0,0, 1,1,1, 0,0,0])
    X_Diagonal   = np.array([1,0,0, 0,1,0, 0,0,1])
    X_Noise1     = np.array([1,0,1, 0,1,0, 1,0,1])
    X_Noise2     = np.array([0,1,0, 1,0,1, 0,1,0])
    X_Noise3     = np.array([0,0,1, 0,1,0, 1,0,0])

    return {
        "垂直線 (1)": X_Vertical,
        "水平線 (1)": X_Horizontal,
        "対角線 (1)": X_Diagonal,
        "ノイズ1 (0)": X_Noise1,
        "ノイズ2 (0)": X_Noise2,
        "ノイズ3 (0)": X_Noise3
    }


# --- 可視化 ---
def visualize_filters_and_tests(W_H, B_H, W_OUT, B_OUT, TEST_PATTERNS):
    H_COUNT = len(B_H)
    PAT_COUNT = len(TEST_PATTERNS)
    PAT_NAMES = list(TEST_PATTERNS.keys())
    PAT_VALUES = list(TEST_PATTERNS.values())

    half = (PAT_COUNT + 1) // 2
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, max(H_COUNT, half), height_ratios=[1, 1, 1])

    # --- 1行目: フィルタ ---
    for i in range(H_COUNT):
        ax = fig.add_subplot(gs[0, i])
        w_3x3 = W_H[i].reshape(3, 3)
        abs_max = np.max(np.abs(w_3x3))
        ax.imshow(w_3x3, cmap='bwr', vmin=-abs_max, vmax=abs_max, aspect='equal')
        ax.set_title(f"H{i+1} (b={B_H[i]:.1f})", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # --- 2,3行目: テストケース ---
    for j in range(PAT_COUNT):
        row = 1 if j < half else 2
        col = j if j < half else j - half
        name = PAT_NAMES[j]
        X = PAT_VALUES[j]
        Y_out, _, _, Y_net = multi_layer_perceptron(X, W_H, B_H, W_OUT, B_OUT)
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(X.reshape(3, 3), cmap='gray', vmin=0, vmax=1, aspect='equal')
        ax.set_title(name, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(1, 2.7, f"中間層出力={float(Y_net)}，最終層出力: {float(Y_out)}", fontsize=10,
                color="black", ha="center")


    plt.tight_layout()
    plt.show()

    # --- 結果表示 ---
    print("\n--- 最終判定結果 ---")
    for name, X in TEST_PATTERNS.items():
        Y_out, _, _, _ = multi_layer_perceptron(X, W_H, B_H, W_OUT, B_OUT)
        print(f"{name}: 判定結果 = {int(Y_out)}")

# ===================================================================
# 共通表示関数（変更なし）
# ===================================================================

def display_transforms(original_img_np, transformed_imgs, titles):
    """元画像と変換された画像を並べて表示する関数"""
    num_cols = len(transformed_imgs) + 1
    fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4))
    
    # 元画像の表示
    axes[0].imshow(original_img_np, cmap='gray')
    axes[0].set_title("元画像", fontsize=12)
    axes[0].axis('off')

    # 変換画像の表示
    for i in range(len(transformed_imgs)):
        ax = axes[i + 1]
        ax.imshow(transformed_imgs[i], cmap='gray')
        ax.set_title(titles[i], fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()