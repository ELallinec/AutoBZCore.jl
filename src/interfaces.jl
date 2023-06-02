"""
    MixedParameters(args::Tuple, kwargs::NamedTuple)

A struct to store full and partial sets of parameters used to evaluate
integrands.
"""
struct MixedParameters{A<:Tuple,K<:NamedTuple}
    args::A
    kwargs::K
end
MixedParameters(args...; kwargs...) = MixedParameters(args, NamedTuple(kwargs))

# parameter fusion and iteration utilities

Base.getindex(p::MixedParameters, i::Int) = getindex(getfield(p, :args), i)
Base.getproperty(p::MixedParameters, name::Symbol) = getproperty(getfield(p, :kwargs), name)

# we only merge non-MixedParameters in the right argument
Base.merge(p::MixedParameters, ::NullParameters) = p
Base.merge(::NullParameters, p::MixedParameters) = p
Base.merge(p::MixedParameters, q) =
    MixedParameters((getfield(p, :args)..., q), getfield(p, :kwargs))
Base.merge(q, p::MixedParameters) =
    MixedParameters((q, getfield(p, :args)...), getfield(p, :kwargs))
Base.merge(p::MixedParameters, q::NamedTuple) =
    MixedParameters(getfield(p, :args), merge(getfield(p, :kwargs), q))
Base.merge(q::NamedTuple, p::MixedParameters) =
    MixedParameters(getfield(p, :args), merge(q, getfield(p, :kwargs)))
Base.merge(p::MixedParameters, q::Tuple) =
    MixedParameters((getfield(p, :args)..., q...), getfield(p, :kwargs))
Base.merge(q::Tuple, p::MixedParameters) =
    MixedParameters((q..., getfield(p, :args)...), getfield(p, :kwargs))
Base.merge(p::MixedParameters, q::MixedParameters) =
    MixedParameters((getfield(p, :args)..., getfield(q, :args)...), merge(getfield(p, :kwargs), getfield(q, :kwargs)))

function paramzip_(::Tuple{}, ::NamedTuple{(), Tuple{}})
    MixedParameters{Tuple{},NamedTuple{(), Tuple{}}}[]
end
function paramzip_(args::Tuple, ::NamedTuple{(), Tuple{}})
    [MixedParameters(arg, NamedTuple()) for arg in zip(args...)]
end
function paramzip_(::Tuple{}, kwargs::NamedTuple)
    [MixedParameters((), NamedTuple{keys(kwargs)}(val)) for val in zip(values(kwargs)...)]
end
function paramzip_(args::Tuple, kwargs::NamedTuple)
    [MixedParameters(arg, NamedTuple{keys(kwargs)}(val)) for (arg, val) in zip(zip(args...), zip(values(kwargs)...))]
end
paramzip(args...; kwargs...) = paramzip_(args, NamedTuple(kwargs))

function paramproduct_(args::Tuple, kwargs::NamedTuple)
    [MixedParameters(item[1:length(args)], NamedTuple{keys(kwargs)}(item[length(args)+1:end])) for item in Iterators.product(args..., values(kwargs)...)]
end
paramproduct(args...; kwargs...) = paramproduct_(args, NamedTuple(kwargs))



"""
    IntegralSolver(prob::IntegralProblem, alg::AbstractAutoBZAlgorithm; abstol, reltol, maxiters, do_inf_transformation)

Constructs a functor that solves the given `IntegralProblem` with the given
`alg` to within the given tolerances. Calling this functor, `fun` with the
syntax `fun(p)` returns the estimated integral `I` after updating `prob` with
the new parameters. Under the hood, this uses the [Integrals.jl
interface](https://docs.sciml.ai/Integrals/stable/) for defining an
`IntegralProblem`, so `f` must be a 2-argument function `f(x,p)`, or if
in-place, a 3-argument function `f(y,x,p)`.

Also, for an `IntegralProblem` with `f` of type [`Integrand`](@ref) type allows
for providing `MixedParameters` via a `fun(args...; kwargs...)` interface so
that the `IntegralSolver` can interface easily with other algorithms, such as
root-finding and interpolation. 
"""
struct IntegralSolver{F,P,iip,T<:IntegralCache{<:IntegralProblem{iip,P,F}}} <: Function
    cache::T
end

function IntegralSolver(prob::IntegralProblem, alg::SciMLBase.AbstractIntegralAlgorithm; kwargs...)
    return IntegralSolver(Integrals.init(prob, alg; kwargs...))
end

# layer to intercept the problem parameters and transform them
remake_problem(_, prob::IntegralProblem, p) = remake(prob, p=p)

remake_problem(prob::IntegralProblem, p) = remake_problem(prob.f, prob, p)


function remake_cache(c, p)
    prob = remake_problem(c.prob, p)
    return IntegralCache(prob, c.alg, c.sensealg, c.kwargs, c.cacheval, c.isfresh)
end

function do_solve(s, p)
    c = remake_cache(s.cache, p)
    return solve!(c)
end

function (s::IntegralSolver)(p=NullParameters())
    sol = do_solve(s, p)
    return sol.u
end


# parallelization

"""
    batchparam(ps, nthreads)

If the cost of a calculation smoothly varies with the parameters `ps`, then
batch `ps` into `nthreads` groups where the `i`th element of group `j` is
`ps[j+(i-1)*nthreads]`
"""
function batchparam(xs, nthreads)
    batches = [Tuple{Int,eltype(xs)}[] for _ in 1:min(nthreads, length(xs))]
    for (i, x) in pairs(IndexStyle(xs), xs)
        push!(batches[mod(i-1, nthreads)+1], (i, x))
    end
    return batches
end

function batcheval(i, p, f, callback)
    t = time()
    sol = do_solve(f, p)
    t = time() - t
    callback(f, i, p, sol, t)
    return sol.u
end


function batchsolve!(out, f, ps, nthreads, callback)
    Threads.@threads for batch in batchparam(ps, nthreads)
        f_ = Threads.threadid() == 1 ? f : deepcopy(f) # avoid data races for in place integrators
        for (i, p) in batch
            out[i] = batcheval(i, p, f_, callback)
        end
    end
    return out
end

solver_type(F, P) = Base.promote_op((f,p) -> do_solve(f,p).u, F, P)

"""
    batchsolve(f::IntegralSolver, ps::AbstractArray, [T]; nthreads=Threads.nthreads())

Evaluate the [`IntegralSolver`](@ref) `f` at each of the parameters `ps` in
parallel. Returns an array similar to `ps` containing the evaluated integrals `I`. This is
a form of multithreaded broadcasting. Providing the return type `f(eltype(ps))::T` is
optional, but will help in case inference of that type fails.
"""
function batchsolve(f::IntegralSolver, ps::AbstractArray, T=solver_type(typeof(f), eltype(ps)); nthreads=Threads.nthreads(), callback=(x...)->nothing)
    return batchsolve!(similar(ps, T), f, ps, nthreads, callback)
end


# provide our own parameter interface for our integrands
abstract type AbstractAutoBZIntegrand{F} end

function remake_problem(f::AbstractAutoBZIntegrand, prob::IntegralProblem, p)
    new = remake(prob, p=merge(prob.p, p))
    return remake_problem(f, new)
end
function evaluate_integrand(f, x, p::MixedParameters)
    return f(x, getfield(p, :args)...; getfield(p, :kwargs)...)
end

function (s::IntegralSolver{<:AbstractAutoBZIntegrand})(args...; kwargs...)
    p = MixedParameters(args...; kwargs...)
    sol = do_solve(s, p)
    return sol.u
end


"""
    Integrand(f, args...; kwargs...)

Represent an integrand with a partial collection of parameters `p`. When the
`Integrand` is invoked with one argument, e.g. `int(x)`, it evaluates `f(x,
p...; kwargs...)`. However when invoked with two arguments, as in an `IntegralProblem`,
e.g. `int(x, p2)`, it evaluates the union of parameters `f(x, p..., p2...; kwargs...)`.
This allows for convenient parametrization of the integrand.
"""
struct Integrand{F,P<:MixedParameters} <: AbstractAutoBZIntegrand{F}
    f::F
    p::P
    Integrand{F}(f::F, p::P) where {F,P<:MixedParameters} = new{F,P}(f, p)
end

function Integrand(f, args...; kwargs...)
    p = MixedParameters(args...; kwargs...)
    return Integrand{typeof(f)}(f, p)
end

# provide Integrals.jl interface
function (f::Integrand)(x, p=NullParameters())
    return evaluate_integrand(f.f, x, merge(f.p, p))
end

function construct_integrand(f::Integrand, iip, p)
    return Integrand(f.f, merge(f.p, p))
end

function remake_problem(f::Integrand, prob::IntegralProblem)
    new = remake(prob, f=Integrand(f.f), p=merge(f.p, prob.p))
    return remake_autobz_problem(f.f, new)
end

"""
    construct_autobz_problem(f, prob)

By dispatching on the type of the user's integrand, `f`, users can `remake` the
`IntegralProblem`. All the parameters 
"""
remake_autobz_problem(_, prob) = prob