import os
from fs_utils import make_dirs
from image_utils import separate_bayer_downsample

def main_bayer_downsample_all():

    # -------------------------------
    # Input / Output dirs
    # -------------------------------
    input_dir = "../../test_images"
    output_dir = "../results/bayer_downsampled"
    make_dirs(output_dir)

    # -------------------------------
    # Settings
    # -------------------------------
    patterns = ["RGGB", "BGGR", "GBRG", "GRBG"]
    save_type = "gray"
    save_mode = "like_c"

    # -------------------------------
    # Process each Bayer pattern
    # -------------------------------
    for pattern in patterns:

        input_path = os.path.join(
            input_dir,
            f"bayer_fullHD_{pattern}.png"
        )

        for color in ("R", "G", "B"):

            outfile = os.path.join(
                output_dir,
                f"{pattern}_{color}_{save_type}_{save_mode}.png"
            )

            separate_bayer_downsample(
                input_path,
                outfile,
                pattern,
                color,
                save_type=save_type,
                save_mode=save_mode
            )

            print(f"Saved → {outfile}")

    print("✨ All downsampled Bayer patterns processed!")


if __name__ == "__main__":
    main_bayer_downsample_all()
