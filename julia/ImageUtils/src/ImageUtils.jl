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
       separate_bayer_downsample,
       make_background,
       make_background_rgb,
       label_components_4conn,
       label_components_8conn

using Images, ProgressMeter, Base.Threads

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

function make_background(pathlist::Vector{String}; mode=:mode)
    println("make_background begins!")
    height, width = size(read_img_gray_float64(pathlist[1]))
    println("height x width = $height x $width")
    if mode == :mean
        background = zeros(Float64, height, width)
        @showprogress desc="Background calculating..." for path in pathlist
            img = read_img_gray_float64(path)
            img_u8 = trunc.(UInt8, img .* 255)   # Cのunsigned charでのキャストと同等
            background .+= Float64.(img_u8)      # 0〜255で加算

            img_u8 = nothing
            img    = nothing
        end
        background ./= length(pathlist)          # 平均 (0〜255)
        background ./= 255.0                     # 0.0〜1.0に正規化
        GC.gc() 
        return background
    end

    if mode == :mode || mode == :median
        votevol = zeros(Int, 256, height, width)
        @showprogress desc="Background calculating..." for path in pathlist
            img = read_img_gray_float64(path)
            img_u8 = trunc.(UInt8, img .* 255)
            @threads for i in 1:width
                @inbounds for j in 1:height
                    votevol[img_u8[j, i]+1, j, i] += 1
                end
            end
        end
        
        if mode == :mode

            argmax_result = argmax(votevol, dims=1) 
            println("argmax_result size is $(size(argmax_result))")
            indices = argmax_result[1, :, :]
            println("indices size is $(size(indices))")
            indices_array = [value[1] for value in indices]
            println("indices_array size is $(size(indices_array))")
            mode_values = indices_array .- 1
            println("mode_values size is $(size(mode_values))")
            background = mode_values ./ 255.0

            return background
        end
        if mode == :median
            # 中央値（偶数枚は中央2ビンの平均）
            N = length(pathlist)
            half_lo = (N + 1) ÷ 2  # 下側中央値の閾値（1-indexedの累積）
            half_hi = (N + 2) ÷ 2  # 上側中央値の閾値（偶数のとき一つ上）

            background = Array{Float64}(undef, height, width)
            @threads for i in 1:width
                @inbounds for j in 1:height
                    csum = 0
                    med_lo = 1
                    med_hi = 1
                    found_lo = false
                    @inbounds for k in 1:256
                        csum += votevol[k, j, i]
                        if !found_lo && csum >= half_lo
                            med_lo = k
                            found_lo = true
                        end
                        if csum >= half_hi
                            med_hi = k
                            break
                        end
                    end
                    med_val = ((med_lo - 1) + (med_hi - 1)) / 2.0  # 0–255の中央値
                    background[j, i] = med_val / 255.0              # 0.0–1.0へ
                end
            end
            return background
        end
    end
end

function make_background_rgb(pathlist::Vector{String}; mode::Symbol=:mode)
    println("make_background_rgb begins!")

    # 1枚目でサイズ取得
    img0 = read_img_rgb_float64(pathlist[1])   # Matrix{RGB{Float64}}
    height, width = size(img0)
    println("height x width = $height x $width")

    if mode == :mean
        backR = zeros(Float64, height, width)
        backG = zeros(Float64, height, width)
        backB = zeros(Float64, height, width)

        @showprogress desc="Background calculating (mean RGB)..." for path in pathlist
            img = read_img_rgb_float64(path)
            cv = channelview(img)  # (3, H, W) in [0,1]

            r_u8 = trunc.(UInt8, cv[1, :, :] .* 255)
            g_u8 = trunc.(UInt8, cv[2, :, :] .* 255)
            b_u8 = trunc.(UInt8, cv[3, :, :] .* 255)

            backR .+= Float64.(r_u8)
            backG .+= Float64.(g_u8)
            backB .+= Float64.(b_u8)

            img = nothing
        end

        backR ./= length(pathlist); backG ./= length(pathlist); backB ./= length(pathlist)
        backR ./= 255.0;            backG ./= 255.0;            backB ./= 255.0
        GC.gc()
        return (R=backR, G=backG, B=backB)
    end

    if mode == :mode || mode == :median
        # votevol[ch][bin, j, i]
        voteR = zeros(Int, 256, height, width)
        voteG = zeros(Int, 256, height, width)
        voteB = zeros(Int, 256, height, width)

        @showprogress desc="Background calculating ($(mode) RGB)..." for path in pathlist
            img = read_img_rgb_float64(path)
            cv = channelview(img)

            r_u8 = trunc.(UInt8, cv[1, :, :] .* 255)
            g_u8 = trunc.(UInt8, cv[2, :, :] .* 255)
            b_u8 = trunc.(UInt8, cv[3, :, :] .* 255)

            @threads for i in 1:width
                @inbounds for j in 1:height
                    voteR[r_u8[j, i] + 1, j, i] += 1
                    voteG[g_u8[j, i] + 1, j, i] += 1
                    voteB[b_u8[j, i] + 1, j, i] += 1
                end
            end

            img = nothing
        end

        if mode == :mode
            # --- R ---
            argmaxR = argmax(voteR, dims=1)
            idxR = argmaxR[1, :, :]
            idxR_arr = [v[1] for v in idxR]
            backR = (idxR_arr .- 1) ./ 255.0

            # --- G ---
            argmaxG = argmax(voteG, dims=1)
            idxG = argmaxG[1, :, :]
            idxG_arr = [v[1] for v in idxG]
            backG = (idxG_arr .- 1) ./ 255.0

            # --- B ---
            argmaxB = argmax(voteB, dims=1)
            idxB = argmaxB[1, :, :]
            idxB_arr = [v[1] for v in idxB]
            backB = (idxB_arr .- 1) ./ 255.0

            return (R=backR, G=backG, B=backB)
        end

        if mode == :median
            N = length(pathlist)
            half_lo = (N + 1) ÷ 2
            half_hi = (N + 2) ÷ 2

            backR = Array{Float64}(undef, height, width)
            backG = Array{Float64}(undef, height, width)
            backB = Array{Float64}(undef, height, width)

            @threads for i in 1:width
                @inbounds for j in 1:height
                    # --- R ---
                    csum = 0; med_lo = 1; med_hi = 1; found_lo = false
                    @inbounds for k in 1:256
                        csum += voteR[k, j, i]
                        if !found_lo && csum >= half_lo
                            med_lo = k; found_lo = true
                        end
                        if csum >= half_hi
                            med_hi = k; break
                        end
                    end
                    backR[j, i] = (((med_lo - 1) + (med_hi - 1)) / 2.0) / 255.0

                    # --- G ---
                    csum = 0; med_lo = 1; med_hi = 1; found_lo = false
                    @inbounds for k in 1:256
                        csum += voteG[k, j, i]
                        if !found_lo && csum >= half_lo
                            med_lo = k; found_lo = true
                        end
                        if csum >= half_hi
                            med_hi = k; break
                        end
                    end
                    backG[j, i] = (((med_lo - 1) + (med_hi - 1)) / 2.0) / 255.0

                    # --- B ---
                    csum = 0; med_lo = 1; med_hi = 1; found_lo = false
                    @inbounds for k in 1:256
                        csum += voteB[k, j, i]
                        if !found_lo && csum >= half_lo
                            med_lo = k; found_lo = true
                        end
                        if csum >= half_hi
                            med_hi = k; break
                        end
                    end
                    backB[j, i] = (((med_lo - 1) + (med_hi - 1)) / 2.0) / 255.0
                end
            end
            return (R=backR, G=backG, B=backB)
        end
    end

    error("Unknown mode: $mode (use :mean, :mode, :median)")
end

function label_components_4conn(bin::AbstractMatrix)
    ny, nx = size(bin)
    lab = zeros(Int, ny, nx)
    boxes = Dict{Int, Tuple{Int,Int,Int,Int,Int}}()  # id => (ymin,ymax,xmin,xmax,area)
    qy = Vector{Int}(undef, ny*nx)
    qx = Vector{Int}(undef, ny*nx)

    cur = 0
    for y in 1:ny, x in 1:nx
        if bin[y,x] && lab[y,x] == 0
            cur += 1
            # BFS開始
            head = 1; tail = 1
            qy[tail] = y; qx[tail] = x

            ymin = y; ymax = y; xmin = x; xmax = x; area = 0
            lab[y,x] = cur

            while head <= tail
                cy = qy[head]; cx = qx[head]; head += 1
                area += 1
                ymin = min(ymin, cy); ymax = max(ymax, cy)
                xmin = min(xmin, cx); xmax = max(xmax, cx)

                # 4近傍
                if cy>1      && bin[cy-1,cx] && lab[cy-1,cx]==0; lab[cy-1,cx]=cur; tail+=1; qy[tail]=cy-1; qx[tail]=cx; end
                if cy<ny     && bin[cy+1,cx] && lab[cy+1,cx]==0; lab[cy+1,cx]=cur; tail+=1; qy[tail]=cy+1; qx[tail]=cx; end
                if cx>1      && bin[cy,cx-1] && lab[cy,cx-1]==0; lab[cy,cx-1]=cur; tail+=1; qy[tail]=cy;   qx[tail]=cx-1; end
                if cx<nx     && bin[cy,cx+1] && lab[cy,cx+1]==0; lab[cy,cx+1]=cur; tail+=1; qy[tail]=cy;   qx[tail]=cx+1; end
            end
            boxes[cur] = (ymin, ymax, xmin, xmax, area)
        end
    end
    return lab, boxes
end
function label_components_8conn(bin::AbstractMatrix)
    ny, nx = size(bin)
    lab = zeros(Int, ny, nx)
    boxes = Dict{Int, Tuple{Int,Int,Int,Int,Int}}()  # id => (ymin,ymax,xmin,xmax,area)
    qy = Vector{Int}(undef, ny*nx)
    qx = Vector{Int}(undef, ny*nx)

    cur = 0
    for y in 1:ny, x in 1:nx
        if bin[y,x] && lab[y,x] == 0
            cur += 1
            # BFS開始
            head = 1; tail = 1
            qy[tail] = y; qx[tail] = x

            ymin = y; ymax = y; xmin = x; xmax = x; area = 0
            lab[y,x] = cur

            while head <= tail
                cy = qy[head]; cx = qx[head]; head += 1
                area += 1
                ymin = min(ymin, cy); ymax = max(ymax, cy)
                xmin = min(xmin, cx); xmax = max(xmax, cx)

                # 8近傍
                for dy in -1:1, dx in -1:1
                    if dy == 0 && dx == 0
                        continue
                    end
                    nyy, nxx = cy+dy, cx+dx
                    if 1 <= nyy <= ny && 1 <= nxx <= nx
                        if bin[nyy,nxx] && lab[nyy,nxx] == 0
                            lab[nyy,nxx] = cur
                            tail += 1
                            qy[tail] = nyy
                            qx[tail] = nxx
                        end
                    end
                end
            end
            boxes[cur] = (ymin, ymax, xmin, xmax, area)
        end
    end
    return lab, boxes
end

end # module