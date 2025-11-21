using Revise
using FSUtils
using ImageUtils

function main_rgb()

    # -------------------------------
    # Input image path
    # -------------------------------
    input_path = "../../test_images/RGB_words_FullHD.png"

    # -------------------------------
    # Output directory
    # -------------------------------
    output_dir = "../results/rgb_separated"
    FSUtils.make_dirs(output_dir)

    # -------------------------------
    # Settings
    # -------------------------------
    order     = :RGB        # :RGB, :BGR, :GBR, :GRB, :BRG, :RBG
    save_type = :rgb        # :gray or :rgb
    save_mode = :like_c     # :standard or :like_c

    # -------------------------------
    # Loop through channels
    # -------------------------------
    for color in (:R, :G, :B)

        # Output file name
        outfile = joinpath(output_dir,
                           "$(color)_$(order)_$(save_type)_$(save_mode).png")

        # Call the universal extractor
        ImageUtils.separate_rgb(input_path, outfile, color;
                     order     = order,
                     save_type = save_type,
                     save_mode = save_mode)

        println("Saved → $(outfile)")
    end

    println("✨ All RGB channels processed!")
end

main_rgb()
