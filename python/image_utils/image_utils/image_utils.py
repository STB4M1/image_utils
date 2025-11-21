from PIL import Image
import numpy as np

def read_img_gray_float64(path: str):
    img = Image.open(path).convert("L")  # グレースケール (0–255)
    out = np.array(img, dtype=np.float64) / 255.0  # Gray{Float64} 相当
    return out

def read_img_rgb_float64(path: str):
    img = Image.open(path).convert("RGB")
    out = np.array(img, dtype=np.float64) / 255.0  # RGB{Float64} 相当
    return out

# ----------------------------------------
# C の unsigned char キャストと完全同等
# ----------------------------------------
def save_gray_like_c(img: np.ndarray, path: str):
    # img : float or int matrix
    # Julia: UInt8.(mod.(Int.(trunc.(img .* 255)), 256))
    v = np.trunc(img * 255).astype(np.int64)    # trunc → Int
    v = np.mod(v, 256).astype(np.uint8)         # mod 256 → UInt8
    Image.fromarray(v, mode="L").save(path)
    return path

# ----------------------------------------
# Julia の標準的な丸め + clamp と完全同等
# ----------------------------------------
def save_gray_standard(img: np.ndarray, path: str):
    # Julia: clamp.(round.(Int, img .* 255), 0, 255)
    v = np.round(img * 255).astype(np.int64)    # round → Int
    v = np.clip(v, 0, 255).astype(np.uint8)     # clamp → UInt8
    Image.fromarray(v, mode="L").save(path)
    return path

# ----------------------------------------------------------
# C の unsigned char キャストと同じ挙動で RGB 保存
# ----------------------------------------------------------
def save_rgb_like_c(R=None, G=None, B=None, path="output.png"):
    # チャンネルリスト
    chans = [c for c in [R, G, B] if c is not None]
    if len(chans) == 0:
        raise ValueError("Specify at least one channel (R, G, B).")

    # サイズ決定（最初のチャンネルに合わせる）
    h, w = chans[0].shape

    # None のチャンネルは 0 埋め
    if R is None: R = np.zeros((h, w), dtype=float)
    if G is None: G = np.zeros((h, w), dtype=float)
    if B is None: B = np.zeros((h, w), dtype=float)

    # Julia: reinterpret.(N0f8, UInt8.(mod.(Int.(trunc(x*255)),256)))
    def to_u8(x):
        v = np.trunc(x * 255).astype(np.int64)   # trunc → Int
        v = np.mod(v, 256).astype(np.uint8)      # mod 256 → UInt8
        return v

    R8 = to_u8(R)
    G8 = to_u8(G)
    B8 = to_u8(B)

    # RGB 画像に結合
    rgb = np.stack([R8, G8, B8], axis=-1)

    Image.fromarray(rgb, mode="RGB").save(path)
    return path

# ----------------------------------------------------------
# Julia の標準的丸め + clamp（RGB）を完全再現
# ----------------------------------------------------------
def save_rgb_standard(R=None, G=None, B=None, path="output.png"):
    # チャンネルリスト
    chans = [c for c in [R, G, B] if c is not None]
    if len(chans) == 0:
        raise ValueError("Specify at least one channel (R, G, B).")

    # サイズ決定
    h, w = chans[0].shape

    # None を 0 配列に
    if R is None: R = np.zeros((h, w), dtype=float)
    if G is None: G = np.zeros((h, w), dtype=float)
    if B is None: B = np.zeros((h, w), dtype=float)

    # Julia: UInt8.(clamp.(round.(Int, x * 255), 0, 255))
    def to_u8(x):
        v = np.round(x * 255).astype(np.int64)       # round → Int
        v = np.clip(v, 0, 255).astype(np.uint8)      # clamp → UInt8
        return v

    R8 = to_u8(R)
    G8 = to_u8(G)
    B8 = to_u8(B)

    rgb = np.stack([R8, G8, B8], axis=-1)

    Image.fromarray(rgb, mode="RGB").save(path)
    return path

# ------------------------------------------------------------
# FULL Julia-compatible separate_rgb
# ------------------------------------------------------------
def separate_rgb(
    in_path: str,
    out_path: str,
    color: str,
    order: str = "RGB",
    save_type: str = "gray",
    save_mode: str = "standard"
):
    # --------------------------
    # Assertions (same as Julia)
    # --------------------------
    assert color in ("R", "G", "B")
    assert save_type in ("gray", "rgb")
    assert save_mode in ("standard", "like_c")
    assert order in ("RGB", "BGR", "GBR", "GRB", "BRG", "RBG")

    # --------------------------------------------------------
    # Load input image → Float64 normalized RGB
    # --------------------------------------------------------
    img_rgb = read_img_rgb_float64(in_path)   # shape (H, W, 3)

    # Extract base channels in R,G,B order
    base_R = img_rgb[:, :, 0]
    base_G = img_rgb[:, :, 1]
    base_B = img_rgb[:, :, 2]

    # --------------------------------------------------------
    # Channel order mapping (Julia と完全同じ)
    # --------------------------------------------------------
    order_map = {
        "RGB": ["R", "G", "B"],
        "BGR": ["B", "G", "R"],
        "GBR": ["G", "B", "R"],
        "GRB": ["G", "R", "B"],
        "BRG": ["B", "R", "G"],
        "RBG": ["R", "B", "G"]
    }

    img_order = order_map[order]     # 例: :BGR → ["B","G","R"]

    # Base channels dictionary
    orig_channels = {
        "R": base_R,
        "G": base_G,
        "B": base_B
    }

    # Channels after applying the order
    ch1 = orig_channels[img_order[0]]
    ch2 = orig_channels[img_order[1]]
    ch3 = orig_channels[img_order[2]]

    # --------------------------------------------------------
    # Pick requested channel in correct order
    # --------------------------------------------------------
    idx = img_order.index(color)  # 0,1,2

    if idx == 0:
        ch = ch1
    elif idx == 1:
        ch = ch2
    else:
        ch = ch3

    # --------------------------------------------------------
    # Save grayscale
    # --------------------------------------------------------
    if save_type == "gray":
        if save_mode == "standard":
            save_gray_standard(ch, out_path)
        else:
            save_gray_like_c(ch, out_path)

    # --------------------------------------------------------
    # Save pseudo-RGB
    # --------------------------------------------------------
    if save_type == "rgb":
        h, w = base_R.shape
        tmpR = np.zeros((h, w), dtype=float)
        tmpG = np.zeros((h, w), dtype=float)
        tmpB = np.zeros((h, w), dtype=float)

        if color == "R":
            tmpR = ch
        elif color == "G":
            tmpG = ch
        elif color == "B":
            tmpB = ch

        if save_mode == "standard":
            save_rgb_standard(R=tmpR, G=tmpG, B=tmpB, path=out_path)
        else:
            save_rgb_like_c(R=tmpR, G=tmpG, B=tmpB, path=out_path)

    print(f"✨ Done → order={order}, type={save_type}, mode={save_mode}, color={color} → {out_path}")

def raw_separate_bayer(bayer: np.ndarray, pattern: str):
    H, W = bayer.shape

    R = np.zeros((H, W), dtype=float)
    G = np.zeros((H, W), dtype=float)
    B = np.zeros((H, W), dtype=float)

    for y in range(H):           # 0-based
        for x in range(W):       # 0-based
            # Julia: yy = y % 2  (Julia y starts at 1 → Python needs +1)
            yy = (y + 1) % 2
            xx = (x + 1) % 2

            v = bayer[y, x]

            if pattern == "RGGB":
                if yy == 1 and xx == 1:
                    R[y, x] = v
                elif yy == 1 and xx == 0:
                    G[y, x] = v
                elif yy == 0 and xx == 1:
                    G[y, x] = v
                else:
                    B[y, x] = v

            elif pattern == "BGGR":
                if yy == 1 and xx == 1:
                    B[y, x] = v
                elif yy == 1 and xx == 0:
                    G[y, x] = v
                elif yy == 0 and xx == 1:
                    G[y, x] = v
                else:
                    R[y, x] = v

            elif pattern == "GBRG":
                if yy == 1 and xx == 1:
                    G[y, x] = v
                elif yy == 1 and xx == 0:
                    B[y, x] = v
                elif yy == 0 and xx == 1:
                    R[y, x] = v
                else:
                    G[y, x] = v

            elif pattern == "GRBG":
                if yy == 1 and xx == 1:
                    G[y, x] = v
                elif yy == 1 and xx == 0:
                    R[y, x] = v
                elif yy == 0 and xx == 1:
                    B[y, x] = v
                else:
                    G[y, x] = v

            else:
                raise ValueError(f"Unknown pattern {pattern}")

    return R, G, B

def separate_bayer(
    in_path: str,
    out_path: str,
    pattern: str,
    channel: str,
    save_type: str = "gray",      # "gray" or "rgb"
    save_mode: str = "standard"   # "standard" or "like_c"
):
    # --------------------------------------------------------
    # Assertions (Julia と完全一致)
    # --------------------------------------------------------
    assert pattern in ("RGGB", "BGGR", "GBRG", "GRBG")
    assert channel in ("R", "G", "B")
    assert save_type in ("gray", "rgb")
    assert save_mode in ("standard", "like_c")

    # --------------------------------------------------------
    # Load Bayer image (Float64 grayscale 0–1)
    # --------------------------------------------------------
    bayer = read_img_gray_float64(in_path)

    # --------------------------------------------------------
    # Split to full-resolution R,G,B channels
    # --------------------------------------------------------
    R, G, B = raw_separate_bayer(bayer, pattern)

    # Select channel
    if channel == "R":
        ch = R
    elif channel == "G":
        ch = G
    else:
        ch = B

    # --------------------------------------------------------
    # Save grayscale
    # --------------------------------------------------------
    if save_type == "gray":
        if save_mode == "standard":
            save_gray_standard(ch, out_path)
        else:
            save_gray_like_c(ch, out_path)

    # --------------------------------------------------------
    # Save pseudo-RGB
    # --------------------------------------------------------
    if save_type == "rgb":
        if save_mode == "standard":
            if channel == "R":
                save_rgb_standard(R=R, path=out_path)
            elif channel == "G":
                save_rgb_standard(G=G, path=out_path)
            else:
                save_rgb_standard(B=B, path=out_path)
        else:  # like_c
            if channel == "R":
                save_rgb_like_c(R=R, path=out_path)
            elif channel == "G":
                save_rgb_like_c(G=G, path=out_path)
            else:
                save_rgb_like_c(B=B, path=out_path)

    print(f"✨ Done Bayer → pattern={pattern}, channel={channel}, type={save_type}, mode={save_mode}")

def separate_bayer_downsample(
    in_path: str,
    out_path: str,
    pattern: str,
    channel: str,
    save_type: str = "gray",
    save_mode: str = "standard"
):
    # --------------------------------------------------------
    # Assertions (Julia と完全一致)
    # --------------------------------------------------------
    assert pattern in ("RGGB", "BGGR", "GBRG", "GRBG")
    assert channel in ("R", "G", "B")
    assert save_type in ("gray", "rgb")
    assert save_mode in ("standard", "like_c")

    # --------------------------------------------------------
    # Load Bayer image
    # --------------------------------------------------------
    bayer = read_img_gray_float64(in_path)
    H, W = bayer.shape

    # --------------------------------------------------------
    # Step 1: Full-resolution Bayer separation
    # --------------------------------------------------------
    R_full, G_full, B_full = raw_separate_bayer(bayer, pattern)

    if channel == "R":
        ch_full = R_full
    elif channel == "G":
        ch_full = G_full
    else:
        ch_full = B_full

    # --------------------------------------------------------
    # Step 2: Downsample (Bayer 有効画素のみ利用)
    # --------------------------------------------------------
    h2 = H // 2
    w2 = W // 2
    out = np.zeros((h2, w2), dtype=float)

    # Julia:
    # yy = (y - 1) ÷ 2 + 1
    # xx = (x - 1) ÷ 2 + 1
    # → Python では 0-based で計算
    for y in range(H):        # y: 0..H-1
        for x in range(W):    # x: 0..W-1
            yy = (y // 2)
            xx = (x // 2)

            # "有効画素だけ" → ch_full[y,x] > 0 のとき代入
            if ch_full[y, x] > 0:
                out[yy, xx] = ch_full[y, x]

    # --------------------------------------------------------
    # Step 3: Save output
    # --------------------------------------------------------
    if save_type == "gray":
        if save_mode == "standard":
            save_gray_standard(out, out_path)
        else:
            save_gray_like_c(out, out_path)

    else:  # save_type == "rgb"
        if save_mode == "standard":
            if channel == "R":
                save_rgb_standard(R=out, path=out_path)
            elif channel == "G":
                save_rgb_standard(G=out, path=out_path)
            else:
                save_rgb_standard(B=out, path=out_path)
        else:  # like_c
            if channel == "R":
                save_rgb_like_c(R=out, path=out_path)
            elif channel == "G":
                save_rgb_like_c(G=out, path=out_path)
            else:
                save_rgb_like_c(B=out, path=out_path)

    print(f"✨ Downsampled Bayer saved → pattern={pattern}, ch={channel}, type={save_type}, mode={save_mode}")
