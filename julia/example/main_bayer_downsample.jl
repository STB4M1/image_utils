using Revise
using FSUtils
using ImageUtils

function main_bayer_downsample_all()

    input_dir = "../../test_images"
    output_dir = "../results/bayer_downsampled"
    FSUtils.make_dirs(output_dir)

    patterns = [:RGGB, :BGGR, :GBRG, :GRBG]

    save_type = :gray
    save_mode = :like_c

    for pattern in patterns

        input_path = joinpath(input_dir, "bayer_fullHD_$(pattern).png")

        for color in (:R, :G, :B)

            outfile = joinpath(output_dir,
                "$(pattern)_$(color)_$(save_type)_$(save_mode).png")

            ImageUtils.separate_bayer_downsample(input_path, outfile, pattern, color;
                                      save_type = save_type,
                                      save_mode = save_mode)

            println("Saved → $(outfile)")
        end
    end

    println("✨ All downsampled Bayer patterns processed!")
end

main_bayer_downsample_all()

