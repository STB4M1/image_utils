module ImageUtils

export read_img_gray_float64,
       read_img_rgb_float64,
       save_gray_like_c,
       save_gray_standard,
       save_rgb_like_c,
       save_rgb_standard,
       separate_rgb,
       raw_separate_bayer,
       separate_bayer,
       separate_bayer_downsample

using Images

function read_img_gray_float64(path::String)
    out = Float64.(Gray.(load(path)))
end

function read_img_rgb_float64(path::String)
    out = RGB{Float64}.(load(path))
end

function save_gray_like_c(img::AbstractMatrix{<:Real}, path::AbstractString)
    img_u8 = UInt8.(mod.(Int.(trunc.(img .* 255)), 256)) # C のキャスト (unsigned char) と同じ挙動
    save(path, img_u8)
    return path
end

function save_gray_standard(img::AbstractMatrix{<:Real}, path::AbstractString)
    img_u8 = clamp.(round.(Int, img .* 255), 0, 255) .|> UInt8
    save(path, img_u8)
    return path
end

function save_rgb_like_c(; R=nothing, G=nothing, B=nothing, path::AbstractString)
    chans = filter(!isnothing, [R, G, B])
    @assert !isempty(chans) "Specify at least one channel (R, G, B)."
    h, w = size(first(chans))
    R, G, B = (isnothing(x) ? zeros(h, w) : x for x in (R, G, B))

    # Convert UInt8 to RGB(N0f8)
    to_u8(x) = reinterpret.(N0f8, UInt8.(mod.(Int.(trunc.(x .* 255)), 256))) # C のキャスト (unsigned char) と同じ挙動

    save(path, colorview(RGB{N0f8}, to_u8(R), to_u8(G), to_u8(B)))
    return path
end

function save_rgb_standard(; R=nothing, G=nothing, B=nothing, path::AbstractString)
    chans = filter(!isnothing, [R, G, B])
    @assert !isempty(chans) "Specify at least one channel (R, G, B)."

    h, w = size(first(chans))
    R, G, B = (isnothing(x) ? zeros(h, w) : x for x in (R, G, B))

    # Standard 0–255 → UInt8 conversion
    to_u8(x) = UInt8.(clamp.(round.(Int, x .* 255), 0, 255))

    save(path, colorview(RGB{N0f8},
        reinterpret.(N0f8, to_u8(R)),
        reinterpret.(N0f8, to_u8(G)),
        reinterpret.(N0f8, to_u8(B))
    ))

    return path
end

function separate_rgb(in_path::String,
                      out_path::String,
                      color::Symbol;
                      order::Symbol = :RGB,          # :RGB, :BGR, :GBR, :GRB, :BRG, :RBG
                      save_type::Symbol = :gray,     # :gray or :rgb
                      save_mode::Symbol = :standard) # :standard or :like_c

    @assert color in (:R, :G, :B)
    @assert save_type in (:gray, :rgb)
    @assert save_mode in (:standard, :like_c)
    @assert order in (:RGB, :BGR, :GBR, :GRB, :BRG, :RBG)

    # --------------------------------------------------------
    # Load input image
    # --------------------------------------------------------
    img_rgb = read_img_rgb_float64(in_path)

    # Base channels from image (always R,G,B order from channelview)
    base_R = channelview(img_rgb)[1, :, :]
    base_G = channelview(img_rgb)[2, :, :]
    base_B = channelview(img_rgb)[3, :, :]

    # --------------------------------------------------------
    # Channel order mapping
    # --------------------------------------------------------
    # Map e.g. :BGR → [:B, :G, :R]
    order_map = Dict(
        :RGB => [:R, :G, :B],
        :BGR => [:B, :G, :R],
        :GBR => [:G, :B, :R],
        :GRB => [:G, :R, :B],
        :BRG => [:B, :R, :G],
        :RBG => [:R, :B, :G]
    )

    # The actual order for this image
    img_order = order_map[order]   # e.g., [:B, :G, :R]

    # Convert symbolic name → actual matrix
    orig_channels = Dict(
        :R => base_R,
        :G => base_G,
        :B => base_B
    )

    # Channels arranged in the image's actual order
    ch1 = orig_channels[img_order[1]]
    ch2 = orig_channels[img_order[2]]
    ch3 = orig_channels[img_order[3]]

    # Pick the requested channel (in correct mapping)
    # find first index where img_order[i] == color
    idx = findfirst(x -> x == color, img_order)
    @assert idx !== nothing "Color not found in channel order mapping."

    ch = idx == 1 ? ch1 : idx == 2 ? ch2 : ch3

    # --------------------------------------------------------
    # Save grayscale
    # --------------------------------------------------------
    if save_type == :gray
        save_mode == :standard ? save_gray_standard(ch, out_path) : save_gray_like_c(ch, out_path)
    end

    # --------------------------------------------------------
    # Save pseudo-RGB
    # --------------------------------------------------------
    if save_type == :rgb
        if save_mode == :standard
            tmpR = zeros(size(base_R))
            tmpG = zeros(size(base_R))
            tmpB = zeros(size(base_R))

            # Set only selected channel
            if color == :R tmpR = ch end
            if color == :G tmpG = ch end
            if color == :B tmpB = ch end

            save_rgb_standard(R=tmpR, G=tmpG, B=tmpB, path=out_path)

        else # like_c
            tmpR = zeros(size(base_R))
            tmpG = zeros(size(base_R))
            tmpB = zeros(size(base_R))

            if color == :R tmpR = ch end
            if color == :G tmpG = ch end
            if color == :B tmpB = ch end

            save_rgb_like_c(R=tmpR, G=tmpG, B=tmpB, path=out_path)
        end
    end

    println("✨ Done → order=$(order), type=$(save_type), mode=$(save_mode), color=$(color) → $(out_path)")
end

function raw_separate_bayer(bayer::Matrix{Float64}, pattern::Symbol)
    H, W = size(bayer)
    R = zeros(Float64, H, W)
    G = zeros(Float64, H, W)
    B = zeros(Float64, H, W)

    for y in 1:H
        for x in 1:W
            yy = y % 2
            xx = x % 2
            v = bayer[y, x]

            if pattern == :RGGB
                if yy==1 && xx==1
                    R[y,x]=v
                elseif yy==1 && xx==0
                    G[y,x]=v
                elseif yy==0 && xx==1
                    G[y,x]=v
                else
                    B[y,x]=v
                end

            elseif pattern == :BGGR
                if yy==1 && xx==1
                    B[y,x]=v
                elseif yy==1 && xx==0
                    G[y,x]=v
                elseif yy==0 && xx==1
                    G[y,x]=v
                else
                    R[y,x]=v
                end

            elseif pattern == :GBRG
                if yy==1 && xx==1
                    G[y,x]=v
                elseif yy==1 && xx==0
                    B[y,x]=v
                elseif yy==0 && xx==1
                    R[y,x]=v
                else
                    G[y,x]=v
                end

            elseif pattern == :GRBG
                if yy==1 && xx==1
                    G[y,x]=v
                elseif yy==1 && xx==0
                    R[y,x]=v
                elseif yy==0 && xx==1
                    B[y,x]=v
                else
                    G[y,x]=v
                end

            else
                error("Unknown pattern $pattern")
            end
        end
    end

    return R, G, B
end

function separate_bayer(in_path::String,
                        out_path::String,
                        pattern::Symbol,
                        channel::Symbol;
                        save_type::Symbol = :gray,      # :gray or :rgb
                        save_mode::Symbol = :standard)  # :standard or :like_c

    @assert pattern in (:RGGB, :BGGR, :GBRG, :GRBG)
    @assert channel in (:R, :G, :B)
    @assert save_type in (:gray, :rgb)
    @assert save_mode in (:standard, :like_c)

    # --------------------------------------------------------
    # Load input Bayer image (Float64 grayscale)
    # --------------------------------------------------------
    bayer = read_img_gray_float64(in_path)

    # --------------------------------------------------------
    # Split into full-resolution R/G/B channels
    # --------------------------------------------------------
    R, G, B = raw_separate_bayer(bayer, pattern)

    # Select channel
    ch = channel == :R ? R : channel == :G ? G : B

    # --------------------------------------------------------
    # Save grayscale
    # --------------------------------------------------------
    if save_type == :gray
        if save_mode == :standard
            save_gray_standard(ch, out_path)
        else
            save_gray_like_c(ch, out_path)
        end
    end

    # --------------------------------------------------------
    # Save pseudo-color RGB
    # --------------------------------------------------------
    if save_type == :rgb
        if save_mode == :standard
            if channel == :R
                save_rgb_standard(R=R, path=out_path)
            elseif channel == :G
                save_rgb_standard(G=G, path=out_path)
            else
                save_rgb_standard(B=B, path=out_path)
            end
        else  # like_c
            if channel == :R
                save_rgb_like_c(R=R, path=out_path)
            elseif channel == :G
                save_rgb_like_c(G=G, path=out_path)
            else
                save_rgb_like_c(B=B, path=out_path)
            end
        end
    end

    println("✨ Done Bayer → pattern=$(pattern), channel=$(channel), type=$(save_type), mode=$(save_mode)")
end

function separate_bayer_downsample(in_path::String,
                                   out_path::String,
                                   pattern::Symbol,
                                   channel::Symbol;
                                   save_type::Symbol = :gray,
                                   save_mode::Symbol = :standard)

    @assert pattern in (:RGGB, :BGGR, :GBRG, :GRBG)
    @assert channel in (:R, :G, :B)
    @assert save_type in (:gray, :rgb)
    @assert save_mode in (:standard, :like_c)

    # --------------------------------------------------------
    # Load Bayer image
    # --------------------------------------------------------
    bayer = read_img_gray_float64(in_path)
    H, W = size(bayer)

    # --------------------------------------------------------
    # Step 1: full-resolution separation
    # --------------------------------------------------------
    R_full, G_full, B_full = raw_separate_bayer(bayer, pattern)

    ch_full = channel == :R ? R_full :
              channel == :G ? G_full : B_full

    # --------------------------------------------------------
    # Step 2: downsample (extract valid Bayer positions only)
    # --------------------------------------------------------
    h2 = div(H, 2)
    w2 = div(W, 2)
    out = zeros(Float64, h2, w2)

    for y in 1:H
        for x in 1:W
            yy = (y - 1) ÷ 2 + 1
            xx = (x - 1) ÷ 2 + 1

            # Only copy pixels that were originally valid for this channel
            if ch_full[y, x] > 0
                out[yy, xx] = ch_full[y, x]
            end
        end
    end

    # --------------------------------------------------------
    # Step 3: Save output
    # --------------------------------------------------------
    if save_type == :gray
        save_mode == :standard ? save_gray_standard(out, out_path) : save_gray_like_c(out, out_path)

    else  # save_type == :rgb
        if save_mode == :standard
            channel == :R && save_rgb_standard(R=out, path=out_path)
            channel == :G && save_rgb_standard(G=out, path=out_path)
            channel == :B && save_rgb_standard(B=out, path=out_path)
        else
            channel == :R && save_rgb_like_c(R=out, path=out_path)
            channel == :G && save_rgb_like_c(G=out, path=out_path)
            channel == :B && save_rgb_like_c(B=out, path=out_path)
        end
    end

    println("✨ Downsampled Bayer saved → pattern=$(pattern), ch=$(channel), type=$(save_type), mode=$(save_mode)")
end

end # module