module GeometryUtils

export padding,
       extract_center,
       load_coefficients,
       quadratic_distortion_correction,
       correct_image

using Images, StatsBase, CUDA

function padding(img::AbstractMatrix{<:Real}, pad_value::Real,
                 width_pad::Int, height_pad::Int)

    H, W = size(img)  # (行=高さ, 列=幅)
    @assert height_pad >= H && width_pad >= W "padサイズが小さすぎます"

    T = eltype(img)
    
    # 入力がGPUならCuArrayを、CPUならArrayを用意
    padded = (isdefined(Main, :CuArray) && isa(img, CuArray)) ?
        CUDA.fill(T(pad_value), height_pad, width_pad) :
        fill(T(pad_value), height_pad, width_pad)

    si = (height_pad - H) ÷ 2
    sj = (width_pad  - W) ÷ 2
    padded[si+1 : si+H, sj+1 : sj+W] .= img

    return copy(padded)  # コピーを返す（必ず新規配列）
end

function extract_center(img::AbstractMatrix, w::Int, h::Int)
    H, W = size(img)
    @assert h <= H && w <= W "指定サイズが元画像より大きいです"

    si = (H - h) ÷ 2
    sj = (W - w) ÷ 2
    return copy(img[si+1:si+h, sj+1:sj+w])  # コピーを返す
end

function load_coefficients(filepath::String)
    coefs = Float64[]
    open(filepath) do file
        for line in eachline(file)
            push!(coefs, parse(Float64, line))
        end
    end
    return coefs
end

function quadratic_distortion_correction(img::Array{<:AbstractFloat,2} , coefa::Vector{<:AbstractFloat})
    @assert size(img)[1] == size(img)[2] "The image must be square. Got $(size(img))."
    @assert length(coefa) == 12 "The coefficients must be 12. Got $(length(coefa))."

    n = size(img)[1]
    bkg = mean(img)
    refX = Array{Int}(undef,n*n)
    refY = Array{Int}(undef,n*n)
    out = Array{Float64}(undef,n,n)
    
    for i in 1:n
        for j in 1:n
            refX[(i-1)*n+j] = Int(round(coefa[1] + coefa[2]*j + coefa[3]*i + coefa[4]*j^2 + coefa[5]*i*j + coefa[6]*i^2))
            refY[(i-1)*n+j] = Int(round(coefa[7] + coefa[8]*j + coefa[9]*i + coefa[10]*j^2 + coefa[11]*i*j + coefa[12]*i^2))
        end
    end

    for i in 1:n
        for j in 1:n
            if (refX[(i-1)*n+j]>=1) && (refX[(i-1)*n+j]<=n) && (refY[(i-1)*n+j]>=1) && (refY[(i-1)*n+j]<=n)
                out[i,j] = img[refY[(i-1)*n+j],refX[(i-1)*n+j]]
            else
                out[i,j] = bkg
            end
        end
    end
    return out 
end

function correct_image(input_image_path::String, output_image_path::String, coefficients_path::String)
    img = Float64.(Gray.(load(input_image_path)))
    coefs = load_coefficients(coefficients_path)
    corrected_img = quadratic_distortion_correction(img, coefs)
    # save(output_image_path, corrected_img)
end

end # module GeometryUtils
