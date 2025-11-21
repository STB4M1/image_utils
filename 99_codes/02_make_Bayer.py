import numpy as np
from PIL import Image
import os
import glob


def make_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Checked/created directory: {d}")

# ============================================================
# 0. Bayer パターン定義
# ============================================================

def bayer_mask(h, w, pattern):
    """
    h, w のサイズで R/G/B の位置を決めるマスクを返す
    pattern: "RGGB", "BGGR", "GBRG", "GRBG"
    """
    R = np.zeros((h, w), dtype=np.float32)
    G = np.zeros((h, w), dtype=np.float32)
    B = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            yy = y % 2
            xx = x % 2

            if pattern == "RGGB":
                if yy == 0 and xx == 0: R[y, x] = 1
                elif yy == 0 and xx == 1: G[y, x] = 1
                elif yy == 1 and xx == 0: G[y, x] = 1
                else: B[y, x] = 1

            elif pattern == "BGGR":
                if yy == 0 and xx == 0: B[y, x] = 1
                elif yy == 0 and xx == 1: G[y, x] = 1
                elif yy == 1 and xx == 0: G[y, x] = 1
                else: R[y, x] = 1

            elif pattern == "GBRG":
                if yy == 0 and xx == 0: G[y, x] = 1
                elif yy == 0 and xx == 1: B[y, x] = 1
                elif yy == 1 and xx == 0: R[y, x] = 1
                else: G[y, x] = 1

            elif pattern == "GRBG":
                if yy == 0 and xx == 0: G[y, x] = 1
                elif yy == 0 and xx == 1: R[y, x] = 1
                elif yy == 1 and xx == 0: B[y, x] = 1
                else: G[y, x] = 1

    return R, G, B


# ============================================================
# 1. 入力画像をロード
# ============================================================

input_path = "../01_images/RGB_words_FullHD.png"
img = np.array(Image.open(input_path)).astype(np.float32) / 255.0

H, W, _ = img.shape
R_ch = img[..., 0]
G_ch = img[..., 1]
B_ch = img[..., 2]


# ============================================================
# 2. Bayer 変換関数
# ============================================================

def make_bayer(img_R, img_G, img_B, pattern, basename="bayer", outdir="."):
    Rmask, Gmask, Bmask = bayer_mask(img_R.shape[0], img_R.shape[1], pattern)

    bayer = img_R * Rmask + img_G * Gmask + img_B * Bmask

    out = (bayer * 255).astype(np.uint8)
    out_img = Image.fromarray(out)

    # ←←★ ここを変更
    filename = os.path.join(outdir, f"{basename}_{pattern}.png")

    out_img.save(filename)
    print("Saved:", filename)


# ============================================================
# 3. 全 Bayer 種類を作る
# ============================================================

output_dir = "../01_images"
make_dirs(output_dir)

patterns = ["RGGB", "BGGR", "GBRG", "GRBG"]

for p in patterns:
    make_bayer(R_ch, G_ch, B_ch, p, basename="bayer_fullHD", outdir=output_dir)
