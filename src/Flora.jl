try Pkg.installed("Revise")
    using Revise
end

__precompile__(true)

module Flora

include("mlp.jl")
include("rnn.jl")

end # module Flora
