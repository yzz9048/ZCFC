import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

def visualize_tsne(features, labels, name, title="t-SNE Visualization"):
    # 标准化特征（可选，根据数据情况决定是否使用）
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 执行t-SNE降维 - 使用 max_iter 代替 n_iter
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    features_2d = tsne.fit_transform(features_scaled)
    
    # 创建可视化
    plt.figure(figsize=(12, 10))
    
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=labels, cmap='tab10', alpha=0.7, s=20)
    
    plt.colorbar(scatter, label='Class')
    plt.title(title, fontsize=16)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.grid(alpha=0.3)
    
    # 添加类别标签
    unique_labels = np.unique(labels)
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        if len(indices) > 0:
            center = np.median(features_2d[indices], axis=0)
            plt.annotate(str(int(label)), center, 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f't-SNE_{name}.png', dpi=300, bbox_inches='tight')
    
    return 