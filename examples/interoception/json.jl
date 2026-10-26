# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Tiny deterministic JSON writer (sorted keys, NaN/Inf -> null) so the benchmark
# needs no extra dependency.

function _json_str(io::IO, s::AbstractString)
    print(io, '"')
    for c in s
        if c == '"' || c == '\\'
            print(io, '\\', c)
        elseif c == '\n'
            print(io, "\\n")
        elseif c == '\t'
            print(io, "\\t")
        elseif c == '\r'
            print(io, "\\r")
        elseif c < ' '
            print(io, "\\u", lpad(string(UInt16(c); base=16), 4, '0'))
        else
            print(io, c)
        end
    end
    print(io, '"')
end

function write_json(io::IO, x, indent::Int=0)
    pad(n) = print(io, "  "^n)
    if x === nothing || x === missing
        print(io, "null")
    elseif x isa Bool
        print(io, x)
    elseif x isa Integer
        print(io, x)
    elseif x isa AbstractFloat
        isfinite(x) ? print(io, x) : print(io, "null")
    elseif x isa Symbol
        _json_str(io, String(x))
    elseif x isa AbstractString
        _json_str(io, x)
    elseif x isa AbstractDict
        ks = sort!(collect(keys(x)); by=string)
        if isempty(ks)
            print(io, "{}")
        else
            println(io, "{")
            for (i, k) in enumerate(ks)
                pad(indent + 1); _json_str(io, string(k)); print(io, ": ")
                write_json(io, x[k], indent + 1)
                println(io, i < length(ks) ? "," : "")
            end
            pad(indent); print(io, "}")
        end
    elseif x isa Union{AbstractVector,Tuple}
        if all(e -> e isa Union{Number,Bool,Nothing,AbstractString,Symbol}, x)
            print(io, "[")
            for (i, e) in enumerate(x)
                write_json(io, e, indent); i < length(x) && print(io, ", ")
            end
            print(io, "]")
        else
            println(io, "[")
            for (i, e) in enumerate(x)
                pad(indent + 1); write_json(io, e, indent + 1)
                println(io, i < length(x) ? "," : "")
            end
            pad(indent); print(io, "]")
        end
    elseif x isa AbstractMatrix
        write_json(io, [collect(r) for r in eachrow(x)], indent)
    elseif x isa NamedTuple
        write_json(io, Dict(string(k) => v for (k, v) in pairs(x)), indent)
    else
        _json_str(io, string(x))
    end
    return nothing
end

json_string(x) = (io = IOBuffer(); write_json(io, x); println(io); String(take!(io)))
