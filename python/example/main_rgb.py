import os
from fs_utils import make_dirs
from image_utils import separate_rgb

def main_rgb():

    # -------------------------------
    # Input image path
    # -------------------------------
    input_path = "../../test_images/RGB_words_FullHD.png"

    # -------------------------------
    # Output directory
    # -------------------------------
    output_dir = "../results/rgb_separated"
    make_dirs(output_dir)

    # -------------------------------
    # Settings
    # -------------------------------
    order     = "RGB"      # "RGB", "BGR", "GBR", "GRB", "BRG", "RBG"
    save_type = "rgb"      # "gray" or "rgb"
    save_mode = "like_c"   # "standard" or "like_c"

    # -------------------------------
    # Loop through channels
    # -------------------------------
    for color in ("R", "G", "B"):

        # Output file name
        outfile = os.path.join(
            output_dir,
            f"{color}_{order}_{save_type}_{save_mode}.png"
        )

        # Call the universal extractor
        separate_rgb(
            input_path,
            outfile,
            color,
            order=order,
            save_type=save_type,
            save_mode=save_mode
        )

        print(f"Saved → {outfile}")

    print("✨ All RGB channels processed!")


if __name__ == "__main__":
    main_rgb()
