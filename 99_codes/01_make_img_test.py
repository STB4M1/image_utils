from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import glob

# ============================================================
# 0. フォント自動検出（Times → Arial → DejaVu）
# ============================================================

def find_font():
    candidates = []

    # Times New Roman
    candidates += glob.glob("/usr/share/fonts/**/Times_New_Roman*.ttf", recursive=True)
    candidates += glob.glob("/usr/share/fonts/**/times*.ttf", recursive=True)

    # Arial
    candidates += glob.glob("/usr/share/fonts/**/Arial*.ttf", recursive=True)
    candidates += glob.glob("/usr/share/fonts/**/arial*.ttf", recursive=True)

    # DejaVu (fallback)
    candidates += glob.glob("/usr/share/fonts/**/DejaVuSans*.ttf", recursive=True)
    candidates += glob.glob("/usr/share/fonts/**/DejaVuSerif*.ttf", recursive=True)

    for f in candidates:
        if os.path.exists(f):
            print("Using font:", f)
            return f

    raise FileNotFoundError("No suitable font found.")

# ============================================================
# 1. 文字切れしない高精度レンダリング
# ============================================================

def render_text_auto(text, font_path, font_size=500, margin=0.20):
    """
    高解像度で文字列を描画し、白文字マスク（0〜1）を返す。
    bbox(top,leftのズレ)を補正して“文字切れ0”で描画する。
    """
    font = ImageFont.truetype(font_path, font_size)

    # 仮キャンバスで bbox を取得
    tmp_img = Image.new("L", (4000, 2000), 0)
    draw_tmp = ImageDraw.Draw(tmp_img)
    bbox = draw_tmp.textbbox((0, 0), text, font=font)

    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # マージン込みでキャンバス作成
    W = int(text_w * (1 + margin * 2))
    H = int(text_h * (1 + margin * 2))

    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)

    # bbox 補正
    pos_x = (W - text_w) // 2 - bbox[0]
    pos_y = (H - text_h) // 2 - bbox[1]

    draw.text((pos_x, pos_y), text, fill=255, font=font)

    return np.array(img, dtype=np.float32) / 255.0

# ============================================================
# 2. キャンバスのサイズを中央に揃える
# ============================================================

def pad_to_center(img, H, W):
    out = np.zeros((H, W), dtype=np.float32)
    h, w = img.shape[:2]
    top = (H - h) // 2
    left = (W - w) // 2
    out[top:top+h, left:left+w] = img
    return out

# ============================================================
# 3. アスペクト比維持の FullHD リサイズ
# ============================================================

def resize_keep_aspect(img, target_w, target_h, bg=0):
    H, W = img.shape[:2]
    scale = min(target_w / W, target_h / H)

    new_w = int(W * scale)
    new_h = int(H * scale)

    img_resized = Image.fromarray(img).resize((new_w, new_h), Image.LANCZOS)
    img_resized = np.array(img_resized)

    out = np.full((target_h, target_w, 3), bg, dtype=img.dtype)

    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2

    out[y_off:y_off+new_h, x_off:x_off+new_w] = img_resized
    return out

# ============================================================
# 4. 四隅に Bayer 2×2 パターンを描画
# ============================================================

def draw_bayer_patterns(img):
    """
    img: (1080, 1920, 3)
    四隅に 2×2 の Bayer マイクロパターンを描画
    """

    RGGB = np.array([
        [[1,0,0], [0,1,0]],
        [[0,1,0], [0,0,1]]
    ], dtype=np.uint8)

    GRBG = np.array([
        [[0,1,0], [1,0,0]],
        [[0,0,1], [0,1,0]]
    ], dtype=np.uint8)

    BGGR = np.array([
        [[0,0,1], [0,1,0]],
        [[0,1,0], [1,0,0]]
    ], dtype=np.uint8)

    GBRG = np.array([
        [[0,1,0], [0,0,1]],
        [[1,0,0], [0,1,0]]
    ], dtype=np.uint8)

    H, W, _ = img.shape

    img[0:2,      0:2]      = RGGB * 255
    img[0:2,      W-2:W]    = GRBG * 255
    img[H-2:H,    0:2]      = BGGR * 255
    img[H-2:H,    W-2:W]    = GBRG * 255

    return img

# ============================================================
# 5. 画像保存のためのフォルダ作成
# ============================================================

def make_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Checked/created directory: {d}")

def save_image(output_dir, filename, img_array):
    make_dirs(output_dir)
    save_path = os.path.join(output_dir, filename)
    Image.fromarray(img_array).save(save_path)
    print(f"Saved: {save_path}")

# ============================================================
# ======================   MAIN 処理   ========================
# ============================================================

def main():

    font_path = find_font()
    print("Final Using Font:", font_path)

    # ---- 文字生成（高精細・文字切れゼロ） ----
    mask_R = render_text_auto("Red", font_path, font_size=500)
    mask_G = render_text_auto("Green", font_path, font_size=500)
    mask_B = render_text_auto("Blue", font_path, font_size=500)

    # ---- キャンバス統一 ----
    H = max(mask_R.shape[0], mask_G.shape[0], mask_B.shape[0])
    W = max(mask_R.shape[1], mask_G.shape[1], mask_B.shape[1])

    mask_R = pad_to_center(mask_R, H, W)
    mask_G = pad_to_center(mask_G, H, W)
    mask_B = pad_to_center(mask_B, H, W)

    # ---- RGB 合成 ----
    img_rgb = np.zeros((H, W, 3), dtype=np.float32)
    img_rgb[..., 0] = mask_R
    img_rgb[..., 1] = mask_G
    img_rgb[..., 2] = mask_B

    # ---- FullHD にアスペクト比維持でリサイズ ----
    img_final = resize_keep_aspect((img_rgb * 255).astype(np.uint8),
                                   1920, 1080)

    # ---- 四隅に Bayer マイクロパターンを描画 ----
    img_final = draw_bayer_patterns(img_final)

    # ---- 保存 ----
    save_image("../01_images", "RGB_words_FullHD.png", img_final)

    print("🎉 完成したよ！")

if __name__ == "__main__":
    main()
