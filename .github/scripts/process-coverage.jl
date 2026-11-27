# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Coverage.jl -> lcov converter for the Windows GPU job. This mirrors
# julia-actions/julia-processcoverage's main.jl, but CI invokes it as
# `julia --pkgimages=no script.jl`: the composite action runs plain `julia`,
# and this host's Application Control policy blocks native pkgimages.
using Pkg

Pkg.activate("coveragetempenv", shared = true)
Pkg.add(PackageSpec(name = "CoverageTools"))

using CoverageTools

dirs = filter!(!isempty, strip.(split(get(ENV, "COVERAGE_DIRECTORIES", "src"), ",")))
for dir in dirs
    isdir(dir) || error("directory \"$dir\" not found!")
end
pfs = mapreduce(process_folder, vcat, dirs)
LCOV.writefile("lcov.info", pfs)
