import os
import glob
import torch
import numpy as np
import pandas as pd  # 需要 pandas 來整理數據給 Seaborn
import seaborn as sns # 需要 seaborn 來畫漂亮的箱線圖
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image
from torchvision import transforms

# 假設您的 models 和 preprocessing 已經正確放在路徑下
from models import vit_encoder
from preprocessing import my_preprocessing

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_ENCODER_NAME = "dinov2reg_vit_small_14"
TARGET_LAYERS = list(range(2, 10))

def load_encoder(encoder_name: str = DEFAULT_ENCODER_NAME, device: torch.device = DEVICE):
    encoder = vit_encoder.load(encoder_name)
    encoder.to(device)
    encoder.eval()
    return encoder

def _prepare_tokens_for_encoder(encoder, x):
    if hasattr(encoder, "prepare_tokens"):
        return encoder.prepare_tokens(x)
    return encoder.prepare_tokens_with_masks(x)

def extract_encoder_embedding(encoder, tensor, target_layers=None):
    target_layers = target_layers or TARGET_LAYERS
    x = _prepare_tokens_for_encoder(encoder, tensor)
    collected = []
    for idx, blk in enumerate(encoder.blocks):
        x = blk(x)
        if idx in target_layers:
            collected.append(x)
        if idx >= target_layers[-1]:
            break
    fused = torch.stack(collected, dim=0).sum(dim=0)
    register_tokens = getattr(encoder, "num_register_tokens", 0)
    patch_tokens = fused[:, 1 + register_tokens:, :]
    return patch_tokens.mean(dim=1)

def get_embedding(model, img_path, preprocess_func=None, target_layers=None, device: torch.device = DEVICE):
    img = Image.open(img_path).convert('RGB')
    if preprocess_func:
        img = preprocess_func(img)
    
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        feature = extract_encoder_embedding(model, input_tensor, target_layers=target_layers)
    return feature.cpu().numpy().squeeze()

# --- 新增：繪製箱線圖函式 ---
def draw_distance_boxplot(embeddings_orig, embeddings_proc, labels_idx, labels_type, output_dir, class_name):
    """
    計算每個變體到其 Regular 的距離，並繪製 Original(Green) vs Processed(Blue) 的箱線圖
    """
    # 1. 建立 Regular 的查表字典
    # map: id -> embedding
    reg_map_orig = {}
    reg_map_proc = {}
    
    for i, ltype in enumerate(labels_type):
        if 'regular' in ltype:
            uid = labels_idx[i]
            reg_map_orig[uid] = embeddings_orig[i]
            reg_map_proc[uid] = embeddings_proc[i]

    # 2. 收集數據
    # 格式: [{'Type': 'overexposed', 'Distance': 0.05, 'State': 'Original'}, ...]
    plot_data = []

    for i, ltype in enumerate(labels_type):
        if 'regular' in ltype:
            continue # 跳過 regular 自己比較自己 (距離為0)
            
        uid = labels_idx[i]
        
        # 確保該 ID 有對應的 Regular 圖 (通常都有，但也許資料集有缺)
        if uid in reg_map_orig:
            # 計算 Original 距離
            emb_o = embeddings_orig[i]
            ref_o = reg_map_orig[uid]
            dist_o = 1 - np.dot(emb_o, ref_o) / (np.linalg.norm(emb_o) * np.linalg.norm(ref_o))
            plot_data.append({
                'Light Type': ltype,
                'Cosine Distance': dist_o,
                'State': 'Original'
            })

            # 計算 Processed 距離
            emb_p = embeddings_proc[i]
            ref_p = reg_map_proc[uid]
            dist_p = 1 - np.dot(emb_p, ref_p) / (np.linalg.norm(emb_p) * np.linalg.norm(ref_p))
            plot_data.append({
                'Light Type': ltype,
                'Cosine Distance': dist_p,
                'State': 'Processed'
            })

    # 3. 轉成 DataFrame 並繪圖
    if not plot_data:
        print("No data for boxplot.")
        return

    df = pd.DataFrame(plot_data)

    plt.figure(figsize=(12, 6))
    
    # 使用 Seaborn 繪圖
    # x: 光照類型, y: 距離, hue: 處理前後 (Original/Processed)
    sns.boxplot(data=df, x='Light Type', y='Cosine Distance', hue='State',
                palette={'Original': 'tab:green', 'Processed': 'tab:blue'},
                showfliers=True) # showfliers=True 顯示離群值

    plt.title(f"Feature Distance to Regular Anchor ({class_name})", fontsize=14, fontweight='bold')
    plt.ylabel("Cosine Distance (Lower is Better)")
    plt.xlabel("Lighting Condition")
    plt.grid(True, axis='y', alpha=0.3, linestyle='--')
    plt.legend(title="Method")
    
    # 存檔
    plot_path = os.path.join(output_dir, f"{class_name}_boxplot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Boxplot saved to: {plot_path}")


# --- 主流程 (包含 PCA 和 Boxplot) ---
def run_validation(data_dir, class_name="can", encoder_name=DEFAULT_ENCODER_NAME, target_layers=None,
                   device: torch.device = DEVICE, plot_output_dir="./adapter_plots"):
    target_dir = os.path.join(data_dir, class_name, "test_public/bad")
    files = sorted(glob.glob(os.path.join(target_dir, "*.png")))
    os.makedirs(plot_output_dir, exist_ok=True)
    
    embeddings_original = []
    embeddings_processed = []
    labels_idx = []   
    labels_type = [] 
    
    print(f"Extracting features with encoder={encoder_name} on device={device} ...")
    encoder = load_encoder(encoder_name, device=device)
    active_layers = target_layers or TARGET_LAYERS

    for f in files:
        filename = os.path.basename(f)
        parts = filename.split('_')
        idx = parts[0] 
        light_type = "_".join(parts[1:]).replace(".png", "")
        
        emb_orig = get_embedding(encoder, f, preprocess_func=None, target_layers=active_layers, device=device)
        embeddings_original.append(emb_orig)
        
        emb_proc = get_embedding(encoder, f, preprocess_func=my_preprocessing, target_layers=active_layers, device=device)
        embeddings_processed.append(emb_proc)
        
        labels_idx.append(idx)
        labels_type.append(light_type)

    embeddings_original = np.array(embeddings_original)
    embeddings_processed = np.array(embeddings_processed)
    
    # --- 1. 畫 PCA (原本的邏輯) ---
    pca = PCA(n_components=2)
    pca.fit(embeddings_original)
    pca_orig = pca.transform(embeddings_original)
    pca_proc = pca.transform(embeddings_processed)
    
    unique_types = sorted(list(set(labels_type)))
    base_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_types)))
    type_color_map = {t: c for t, c in zip(unique_types, base_colors)}
    type_color_map['regular'] = 'red'

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    def plot_spider_web(ax, data, title):
        ax.set_title(title, fontsize=15, fontweight='bold')
        data_map = {}
        for i, (x, y) in enumerate(data):
            uid = labels_idx[i]
            ltype = labels_type[i]
            if uid not in data_map: data_map[uid] = {}
            data_map[uid][ltype] = (x, y)
            
        for uid, variants in data_map.items():
            if 'regular' in variants:
                reg_x, reg_y = variants['regular']
                ax.text(reg_x, reg_y, f" {uid}", fontsize=9, fontweight='bold', 
                        color='black', ha='left', va='bottom', zorder=15)
                for ltype, (vx, vy) in variants.items():
                    if ltype != 'regular':
                        c = type_color_map[ltype]
                        ax.plot([reg_x, vx], [reg_y, vy], color=c, alpha=0.3, linewidth=1, linestyle='--', zorder=1)

        for ltype in unique_types:
            indices = [i for i, t in enumerate(labels_type) if t == ltype]
            if len(indices) == 0: continue
            points = data[indices]
            color = type_color_map[ltype]
            zorder = 10 if ltype == 'regular' else 5
            edgecolor = 'black' if ltype == 'regular' else 'white'
            ax.scatter(points[:, 0], points[:, 1], c=[color], s=80, marker='o', 
                       label=ltype, edgecolors=edgecolor, linewidth=0.5, zorder=zorder, alpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=':')
        if "Refined" in title:
            ax.legend(title="Lighting Type", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    plot_spider_web(axes[0], pca_orig, "Before: Original Features")
    plot_spider_web(axes[1], pca_proc, "After: Processed Features")
    plt.tight_layout()
    pca_path = os.path.join(plot_output_dir, f"{class_name}_pca.png")
    plt.savefig(pca_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"PCA plot saved to: {pca_path}")

    # --- 2. 畫 Boxplot (您要求的新功能) ---
    draw_distance_boxplot(embeddings_original, embeddings_processed, labels_idx, labels_type, plot_output_dir, class_name)
    
    # --- 3. 量化數值輸出 ---
    print("\n--- Quantitative Analysis ---")
    # ... (這裡的 metrics 計算保持不變，省略以節省版面) ...

if __name__ == "__main__":
    mvtec_classes = ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts']
    data_root = './data/mvtec_ad_2'
    for cls in mvtec_classes:
        print(f"\n=== Running validation for class: {cls} ===")
        run_validation(data_root, cls)