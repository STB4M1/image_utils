module FSUtils

export make_dirs

"""
    make_dirs(dirs...)

Make directories if not existing.
"""
function make_dirs(dirs...)
    for dir in dirs
        mkpath(dir)
        println("Checked/created directory: $dir")
    end
end

end # module
