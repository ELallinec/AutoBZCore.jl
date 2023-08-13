# Methods an algorithm must define
# - init_cacheval
# - do_solve


# Here we replicate the algorithms provided by SciML

"""
    QuadGKJL(; order = 7, norm = norm)

Duplicate of the QuadGKJL provided by Integrals.jl.
"""
struct QuadGKJL{F} <: IntegralAlgorithm
    order::Int
    norm::F
end
function QuadGKJL(; order = 7, norm = norm)
    return QuadGKJL(order, norm)
end

function init_segbuf(f, dom, p, norm)
    a, b = endpoints(dom)
    x, s = (a+b)/2, (b-a)/2
    TX = typeof(x)
    fx_s = f(x, p) * s
    TI = typeof(fx_s)
    TE = typeof(norm(fx_s))
    return IteratedIntegration.alloc_segbuf(TX, TI, TE)
end
function init_segbuf(f::InplaceIntegrand, dom, p, norm)
    a, b = endpoints(dom)
    x, s = (a+b)/2, (b-a)/2
    TX = typeof(x)
    TI = typeof(f.I)
    fill!(f.I, zero(eltype(f.I)))
    TE = typeof(norm(f.I))
    return IteratedIntegration.alloc_segbuf(TX, TI, TE)
end
function init_segbuf(f::BatchIntegrand, dom, p, norm)
    a, b = endpoints(dom)
    x, s = (a+b)/2, (b-a)/2
    TX = typeof(x)
    fx_s = zero(eltype(f.y)) * s    # TODO BatchIntegrand(InplaceIntegrand) should depend on size of result
    TI = typeof(fx_s)
    TE = typeof(norm(fx_s))
    return IteratedIntegration.alloc_segbuf(TX, TI, TE)
end
function init_cacheval(f, dom, p, alg::QuadGKJL)
    f isa BatchIntegrand && throw(ArgumentError("QuadGK doesn't support batched integrands"))
    return init_segbuf(f, dom, p, alg.norm)
end

function do_solve(f::F, dom, p, alg::QuadGKJL, cacheval;
                    reltol = nothing, abstol = nothing, maxiters = typemax(Int)) where {F}

    segs = segments(dom)
    if f isa InplaceIntegrand
        g! = (y, x) -> f.f!(y, x, p)
        val, err = quadgk!(g!, f.I, segs..., maxevals = maxiters,
                        rtol = reltol, atol = abstol, order = alg.order, norm = alg.norm, segbuf=cacheval)
        return IntegralSolution(val, err, true)
    else
        g = x -> f(x, p)
        val, err = quadgk(g, segs..., maxevals = maxiters,
                        rtol = reltol, atol = abstol, order = alg.order, norm = alg.norm, segbuf=cacheval)
        return IntegralSolution(val, err, true)
    end
end

"""
    HCubatureJL(; norm=norm, initdiv=1)

A copy of `HCubatureJL` from Integrals.jl.
"""
struct HCubatureJL{N} <: IntegralAlgorithm
    norm::N
    initdiv::Int
end
HCubatureJL(; norm=norm, initdiv=1) = HCubatureJL(norm, initdiv)

function init_cacheval(f, dom, p, ::HCubatureJL)
    f isa BatchIntegrand && throw(ArgumentError("HCubature doesn't support batching"))
    return Some(nothing)
end

function do_solve(f, dom, p, alg::HCubatureJL, cacheval;
                    reltol = 0, abstol = 0, maxiters = typemax(Int))

    g = if f isa InplaceIntegrand
        fx = f.I / oneunit(eltype(dom))
        x -> (f.f!(fx, x, p); fx*one(eltype(dom)))
    else
        x -> f(x, p)
    end
    a, b = endpoints(dom)
    routine = a isa Number ? hquadrature : hcubature
    val, err = routine(g, a, b; norm = alg.norm, initdiv = alg.initdiv, atol=abstol, rtol=reltol, maxevals=maxiters)
    return IntegralSolution(val, err, true)
end

"""
    trapz(n::Integer)

Return the weights and nodes on the standard interval [-1,1] of the [trapezoidal
rule](https://en.wikipedia.org/wiki/Trapezoidal_rule).
"""
function trapz(n::Integer)
    @assert n > 1
    r = range(-1, 1, length=n)
    x = collect(r)
    halfh = step(r)/2
    h = step(r)
    w = [ (i == 1) || (i == n) ? halfh : h for i in 1:n ]
    return (x, w)
end

"""
    QuadratureFunction(; fun=trapz, npt=50, nthreads=Threads.nthreads())

Quadrature rule for the standard interval [-1,1] computed from a function `x, w = fun(npt)`.
The nodes and weights should be set so the integral of `f` on [-1,1] is `sum(w .* f.(x))`.
The default quadrature rule is [`trapz`](@ref), although other packages provide rules, e.g.

    using FastGaussQuadrature
    alg = QuadratureFunction(fun=gausslegendre, npt=100)

`nthreads` sets the numbers of threads used to parallelize the quadrature only when the
integrand is a [`BatchIntegrand`](@ref), in which case the user must parallelize the
integrand evaluations. For no threading set `nthreads=1`.
"""
struct QuadratureFunction{F} <: IntegralAlgorithm
    fun::F
    npt::Int
    nthreads::Int
end
QuadratureFunction(; fun=trapz, npt=50, nthreads=Threads.nthreads()) = QuadratureFunction(fun, npt, nthreads)
init_buffer(f, len) = nothing
init_buffer(f::BatchIntegrand, len) = Vector{eltype(f.y)}(undef, len)
function init_cacheval(f, dom::PuncturedInterval, p, alg::QuadratureFunction)
    buf = init_buffer(f, alg.nthreads)
    x, w = alg.fun(alg.npt)
    return (rule=[(w,x) for (w,x) in zip(w,x)], buffer=buf)
end

function do_solve(f, dom, p, alg::QuadratureFunction, cacheval;
                    reltol = nothing, abstol = nothing, maxiters = typemax(Int))
    rule = cacheval.rule; buffer=cacheval.buffer
    segs = segments(dom)
    g = if f isa BatchIntegrand
        xx = eltype(f.x) === Nothing ? typeof((segs[1]+segs[end])/2)[] : f.x
        AutoSymPTR.BatchIntegrand((y,x) -> f.f!(y,x,p), f.y, xx, max_batch=f.max_batch)
    elseif f isa InplaceIntegrand
        AutoSymPTR.InplaceIntegrand((y,x) -> f.f!(y,x,p), f.I)
    else
        x -> f(x, p)
    end
    I = sum(1:length(segs)-1) do i
        a, b = segs[i], segs[i+1]
        s = (b-a)/2
        rule = AutoSymPTR.AffineQuad(rule, s, a, 1, s)
        return AutoSymPTR.quadsum(rule, g, s, buffer)
    end

    return IntegralSolution(I, nothing, true)
end

# Here we put the quadrature algorithms from IteratedIntegration

"""
    AuxQuadGKJL(; order = 7, norm = norm)

Generalization of the QuadGKJL provided by Integrals.jl that allows for `AuxValue`d
integrands for auxiliary integration and multi-threaded evaluation with the `batch` argument
to `IntegralProblem`
"""
struct AuxQuadGKJL{F} <: IntegralAlgorithm
    order::Int
    norm::F
end
function AuxQuadGKJL(; order = 7, norm = norm)
    return AuxQuadGKJL(order, norm)
end

function init_cacheval(f, dom, p, alg::AuxQuadGKJL)
    return init_segbuf(f, dom, p, alg.norm)
end

function do_solve(f, dom, p, alg::AuxQuadGKJL, cacheval;
                    reltol = nothing, abstol = nothing, maxiters = typemax(Int))

    segs = segments(dom)
    if f isa InplaceIntegrand
        g! = (y, x) -> f.f!(y, x, p)
        val, err = auxquadgk!(g!, f.I, segs, maxevals = maxiters,
                        rtol = reltol, atol = abstol, order = alg.order, norm = alg.norm, segbuf=cacheval)
        return IntegralSolution(val, err, true)
    elseif f isa BatchIntegrand
        xx = eltype(f.x) === Nothing ? typeof((segs[1]+segs[end])/2)[] : f.x
        g = IteratedIntegration.BatchIntegrand((y, x) -> f.f!(y, x, p), f.y, xx, max_batch=f.max_batch)
        val, err = auxquadgk(g, segs, maxevals = maxiters,
                        rtol = reltol, atol = abstol, order = alg.order, norm = alg.norm, segbuf=cacheval)
        return IntegralSolution(val, err, true)
    else
        g = x -> f(x, p)
        val, err = auxquadgk(g, segs, maxevals = maxiters,
                        rtol = reltol, atol = abstol, order = alg.order, norm = alg.norm, segbuf=cacheval)
        return IntegralSolution(val, err, true)
    end
end

"""
    ContQuadGKJL(; order = 7, norm = norm, rho = 1.0, rootmeth = IteratedIntegration.ContQuadGK.NewtonDeflation())

A 1d contour deformation quadrature scheme for scalar, complex-valued integrands. It
defaults to regular `quadgk` behavior on the real axis, but if it finds a root of 1/f
nearby, in the sense of Bernstein ellipse for the standard segment `[-1,1]` with semiaxes
`cosh(rho)` and `sinh(rho)`, on either the upper/lower half planes, then it dents the
contour away from the presumable pole.
"""
struct ContQuadGKJL{F,M} <: IntegralAlgorithm
    order::Int
    norm::F
    rho::Float64
    rootmeth::M
end
function ContQuadGKJL(; order = 7, norm = norm, rho = 1.0, rootmeth = IteratedIntegration.ContQuadGK.NewtonDeflation())
    return ContQuadGKJL(order, norm, rho, rootmeth)
end

function init_cacheval(f, dom, p, alg::ContQuadGKJL)
    f isa BatchIntegrand && throw(ArgumentError("ContQuadGK doesn't support batching"))
    f isa InplaceIntegrand && throw(ArgumentError("ContQuadGK doesn't support inplace integrands"))

    a, b = endpoints(dom)
    x, s = (a+b)/2, (b-a)/2
    TX = typeof(x)
    fx_s = one(ComplexF64) * s # currently the integrand is forcibly written to a ComplexF64 buffer
    TI = typeof(fx_s)
    TE = typeof(alg.norm(fx_s))
    r_segbuf = IteratedIntegration.ContQuadGK.PoleSegment{TX,TI,TE}[]
    fc_s = f(complex(x), p) * complex(s) # the regular evalrule is used on complex segments
    TCX = typeof(complex(x))
    TCI = typeof(fc_s)
    TCE = typeof(alg.norm(fc_s))
    c_segbuf = IteratedIntegration.ContQuadGK.Segment{TCX,TCI,TCE}[]
    return (r=r_segbuf, c=c_segbuf)
end

function do_solve(f, dom, p, alg::ContQuadGKJL, cacheval;
                    reltol = nothing, abstol = nothing, maxiters = typemax(Int))

    segs = segments(dom)
    g = x -> f(x, p)
    val, err = contquadgk(g, segs, maxevals = maxiters, rho = alg.rho, rootmeth = alg.rootmeth,
                    rtol = reltol, atol = abstol, order = alg.order, norm = alg.norm, r_segbuf=cacheval.r, c_segbuf=cacheval.c)
    return IntegralSolution(val, err, true)
end

"""
    MeroQuadGKJL(; order = 7, norm = norm, rho = 1.0, rootmeth = IteratedIntegration.MeroQuadGK.NewtonDeflation())

A 1d pole subtraction quadrature scheme for scalar, complex-valued integrands that are
meromorphic. It defaults to regular `quadgk` behavior on the real axis, but if it finds
nearby roots of 1/f, in the sense of Bernstein ellipse for the standard segment `[-1,1]`
with semiaxes `cosh(rho)` and `sinh(rho)`, it attempts pole subtraction on that segment.
"""
struct MeroQuadGKJL{F,M} <: IntegralAlgorithm
    order::Int
    norm::F
    rho::Float64
    rootmeth::M
end
function MeroQuadGKJL(; order = 7, norm = norm, rho = 1.0, rootmeth = IteratedIntegration.MeroQuadGK.NewtonDeflation())
    return MeroQuadGKJL(order, norm, rho, rootmeth)
end

function init_cacheval(f, dom, p, alg::MeroQuadGKJL)
    f isa BatchIntegrand && throw(ArgumentError("MeroQuadGK doesn't support batching"))
    f isa InplaceIntegrand && throw(ArgumentError("MeroQuadGK doesn't support inplace integrands"))
    a, b = endpoints(dom)
    x, s = (a + b)/2, (b-a)/2
    fx_s = one(ComplexF64) * s # ignore the actual integrand since it is written to CF64 array
    err = alg.norm(fx_s)
    return IteratedIntegration.alloc_segbuf(typeof(x), typeof(fx_s), typeof(err))
end

function do_solve(f, dom, p, alg::MeroQuadGKJL, cacheval;
                    reltol = nothing, abstol = nothing, maxiters = typemax(Int))

    segs = segments(dom)
    g = x -> f(x, p)
    val, err = meroquadgk(g, segs, maxevals = maxiters, rho = alg.rho, rootmeth = alg.rootmeth,
                    rtol = reltol, atol = abstol, order = alg.order, norm = alg.norm, segbuf=cacheval)
    return IntegralSolution(val, err, true)
end

# Algorithms from AutoSymPTR.jl

"""
    MonkhorstPack(; npt=50, syms=nothing, nthreads=Threads.nthreads())

Periodic trapezoidal rule with a fixed number of k-points per dimension, `npt`,
using the `PTR` rule from [AutoSymPTR.jl](https://github.com/lxvm/AutoSymPTR.jl).
`nthreads` sets the numbers of threads used to parallelize the quadrature only when the
integrand is a [`BatchIntegrand`](@ref), in which case the user must parallelize the
integrand evaluations. For no threading set `nthreads=1`.
**The caller should check that the integral is converged w.r.t. `npt`**.
"""
struct MonkhorstPack{S} <: IntegralAlgorithm
    npt::Int
    syms::S
    nthreads::Int
end
MonkhorstPack(; npt=50, syms=nothing, nthreads=Threads.nthreads()) = MonkhorstPack(npt, syms, nthreads)
function init_rule(dom::Basis, alg::MonkhorstPack)
    # rule = AutoSymPTR.MonkhorstPackRule(alg.syms, alg.a, alg.nmin, alg.nmax, alg.n₀, alg.Δn)
    # return rule(eltype(dom), Val(ndims(dom)))
    if alg.syms === nothing
        return AutoSymPTR.PTR(eltype(dom), Val(ndims(dom)), alg.npt)
    else
        return AutoSymPTR.MonkhorstPack(eltype(dom), Val(ndims(dom)), alg.npt, alg.syms)
    end
end

rule_type(::AutoSymPTR.PTR{N,T}) where {N,T} = SVector{N,T}
rule_type(::AutoSymPTR.MonkhorstPack{N,T}) where {N,T} = SVector{N,T}

function init_cacheval(f, dom::Basis, p, alg::MonkhorstPack)
    rule = init_rule(dom, alg)
    buf = init_buffer(f, alg.nthreads)
    return (rule=rule, buffer=buf)
end

function do_solve(f, dom, p, alg::MonkhorstPack, cacheval;
                    reltol = nothing, abstol = nothing, maxiters = typemax(Int))
    g = if f isa BatchIntegrand
        xx = eltype(f.x) === Nothing ? typeof(dom*zero(rule_type(cacheval.rule)))[] : f.x
        AutoSymPTR.BatchIntegrand((y,x) -> f.f!(y,x,p), f.y, xx, max_batch=f.max_batch)
    elseif f isa InplaceIntegrand
        AutoSymPTR.InplaceIntegrand((y,x) -> f.f!(y,x,p), f.I)
    else
        x -> f(x, p)
    end
    I = cacheval.rule(g, dom, cacheval.buffer)
    return IntegralSolution(I, nothing, true)
end

"""
    AutoSymPTRJL(; norm=norm, a=1.0, nmin=50, nmax=1000, n₀=6, Δn=log(10), keepmost=2, nthreads=Threads.nthreads())

Periodic trapezoidal rule with automatic convergence to tolerances passed to the
solver with respect to `norm` using the routine `autosymptr` from
[AutoSymPTR.jl](https://github.com/lxvm/AutoSymPTR.jl).
`nthreads` sets the numbers of threads used to parallelize the quadrature only when the
integrand is a [`BatchIntegrand`](@ref), in which case the user must parallelize the
integrand evaluations. For no threading set `nthreads=1`.
**This algorithm is the most efficient for smooth integrands**.
"""
struct AutoSymPTRJL{F,S} <: IntegralAlgorithm
    norm::F
    a::Float64
    nmin::Int
    nmax::Int
    n₀::Float64
    Δn::Float64
    keepmost::Int
    syms::S
    nthreads::Int
end
function AutoSymPTRJL(; norm=norm, a=1.0, nmin=50, nmax=1000, n₀=6.0, Δn=log(10), keepmost=2, syms=nothing, nthreads=Threads.nthreads())
    return AutoSymPTRJL(norm, a, nmin, nmax, n₀, Δn, keepmost, syms, nthreads)
end
function init_rule(dom::Basis, alg::AutoSymPTRJL)
    return AutoSymPTR.MonkhorstPackRule(alg.syms, alg.a, alg.nmin, alg.nmax, alg.n₀, alg.Δn)
end
function init_cacheval(f, dom::Basis, p, alg::AutoSymPTRJL)
    rule = init_rule(dom, alg)
    cache = AutoSymPTR.alloc_cache(eltype(dom), Val(ndims(dom)), rule)
    buffer = init_buffer(f, alg.nthreads)
    return (rule=rule, cache=cache, buffer=buffer)
end

function do_solve(f, dom, p, alg::AutoSymPTRJL, cacheval;
                    reltol = nothing, abstol = nothing, maxiters = typemax(Int))

    g = if f isa BatchIntegrand
        xx = eltype(f.x) === Nothing ? typeof(dom*zero(rule_type(cacheval.cache[1])))[] : f.x
        AutoSymPTR.BatchIntegrand((y,x) -> f.f!(y,x,p), f.y, xx, max_batch=f.max_batch)
    elseif f isa InplaceIntegrand
        AutoSymPTR.InplaceIntegrand((y,x) -> f.f!(y,x,p), f.I)
    else
        x -> f(x, p)
    end
    val, err = autosymptr(g, dom; syms = alg.syms, rule = cacheval.rule, cache = cacheval.cache, keepmost = alg.keepmost,
        abstol = abstol, reltol = reltol, maxevals = maxiters, norm=alg.norm, buffer=cacheval.buffer)
    return IntegralSolution(val, err, true)
end

"""
    NestedQuad(alg::IntegralAlgorithm)
    NestedQuad(algs::IntegralAlgorithm...)

Nested integration by repeating one quadrature algorithm or composing a list of algorithms.
The domain of integration must be an `AbstractIteratedLimits` from the
IteratedIntegration.jl package. Analogous to `nested_quad` from IteratedIntegration.jl.
The integrand should expect `SVector` inputs. Do not use this for very high-dimensional
integrals, since the compilation time scales very poorly with respect to dimensionality.
In order to improve the compilation time, FunctionWrappers.jl is used to enforce type
stability of the integrand, so you should always pick the widest integration limit type so
that inference works properly. For example, if [`ContQuadGKJL`](@ref) is used as an
algorithm in the nested scheme, then the limits of integration should be made complex.
"""
struct NestedQuad{T} <: IntegralAlgorithm
    algs::T
    NestedQuad(alg::IntegralAlgorithm) = new{typeof(alg)}(alg)
    NestedQuad(algs::Tuple{Vararg{IntegralAlgorithm}}) = new{typeof(algs)}(algs)
end
NestedQuad(algs::IntegralAlgorithm...) = NestedQuad(algs)

function nested_cacheval(f::F, p::P, algs, segs, lims, state, x, xs...) where {F,P}
    dom = PuncturedInterval(segs)
    a, b = segs[1], segs[2]
    dim = ndims(lims)
    alg = algs[dim]
    mid = (a+b)/2 # sample point that should be safe to evaluate
    next = limit_iterate(lims, state, mid) # see what the next limit gives
    if xs isa Tuple{} # the next integral takes us to the inner integral
        # base case test integrand of the inner integral
        # we need to pass dummy integrands to all the outer integrals so that they can build
        # their caches with the right types
        if f isa BatchIntegrand
            # Batch integrate the inner integral only
            cacheval = init_cacheval(BatchIntegrand(nothing, f.y, f.x, max_batch=f.max_batch), dom, p, alg)
            return ((cacheval, oneunit(eltype(f.y))*mid),)
        elseif f isa InplaceIntegrand
            # Inplace integrate through the whole nest structure
            fxi = f.I*mid/oneunit(prod(next))
            cacheval = init_cacheval(InplaceIntegrand(nothing, fxi), dom, p, alg)
            return ((cacheval, fxi),)
        else
            fx = f(next,p)
            cacheval = init_cacheval((x, p) -> fx, dom, p, alg)
            return ((cacheval, fx*mid),)
        end
    else
        algs_ = algs[1:dim-1]
        nest = nested_cacheval(f, p, algs_, next..., x, xs[1:dim-2]...)
        h = nest[end][2]
        hx = h*mid
        # units may change for outer integral
        if f isa InplaceIntegrand
            cacheval = init_cacheval(InplaceIntegrand(nothing, hx), dom, p, alg)
            return (nest..., (cacheval, hx))
        else
            cacheval = init_cacheval((x, p) -> h, dom, p, alg)
            return (nest..., (cacheval, hx))
        end
    end
end
function init_cacheval(f, dom::AbstractIteratedLimits, p, alg::NestedQuad)
    algs = alg.algs isa IntegralAlgorithm ? ntuple(i -> alg.algs, Val(ndims(dom))) : alg.algs
    return nested_cacheval(f, p, algs, limit_iterate(dom)..., interior_point(dom)...)
end

function init_nest(f::F, fxx, dom, p,lims, state, algs, cacheval; kws_...) where {F}
    kws = NamedTuple(kws_)
    FX = typeof(fxx/oneunit(eltype(dom)))
    TX = eltype(dom)
    TP = Tuple{typeof(p),typeof(lims),typeof(state)}
    if algs isa Tuple{} # inner integral
        if f isa BatchIntegrand
            return f
        elseif f isa InplaceIntegrand
            return InplaceIntegrand(FunctionWrapper{Nothing,Tuple{FX,TX,TP}}() do y, x, (p, lims, state)
                f.f!(y, limit_iterate(lims, state, x), p)
                return nothing
            end, f.I)
        else
            return FunctionWrapper{FX,Tuple{TX,TP}}() do x, (p, lims, state)
                return f(limit_iterate(lims, state, x), p)
            end
        end
    else
        if f isa InplaceIntegrand
            return InplaceIntegrand(FunctionWrapper{Nothing,Tuple{FX,TX,TP}}() do y, x, (p, lims, state)
                segs, lims_, state_ = limit_iterate(lims, state, x)
                len = segs[end] - segs[1]
                kwargs = haskey(kws, :abstol) ? merge(kws, (abstol=kws.abstol/len,)) : kws
                sol = do_solve(InplaceIntegrand(f.f!, f.I / oneunit(x)), lims_, NestState(p, segs, state_), NestedQuad(algs), cacheval; kwargs...)
                y .= sol.u
                return nothing
            end, f.I)
        else
            return FunctionWrapper{FX,Tuple{TX,TP}}() do x, (p, lims, state)
                segs, lims_, state_ = limit_iterate(lims, state, x)
                len = segs[end] - segs[1]
                kwargs = haskey(kws, :abstol) ? merge(kws, (abstol=kws.abstol/len,)) : kws
                sol = do_solve(f, lims_, NestState(p, segs, state_), NestedQuad(algs), cacheval; kwargs...)
                return sol.u
            end
        end
    end
end

struct NestState{P,G,S}
    p::P
    segs::G
    state::S
end

function do_solve(f::F, lims::AbstractIteratedLimits, p_, alg::NestedQuad, cacheval; kws...) where {F}
    g, p, segs, state = if p_ isa NestState
        f, p_.p, p_.segs, p_.state
    else
        gg = if f isa BatchIntegrand
            fx = eltype(f.x) === Nothing ? typeof(interior_point(lims))[] : f.x
            BatchIntegrand(f.y, similar(f.x, eltype(eltype(fx))), max_batch=f.max_batch) do y, xs, (p, lims, state)
                resize!(fx, length(xs))
                f.f!(y, map!(x -> limit_iterate(lims, state, x), fx, xs), p)
            end
        else
            f
        end
        seg, lim, sta = limit_iterate(lims)
        gg, p_, seg, sta
    end
    dom = PuncturedInterval(segs)
    dim = ndims(lims) # constant propagation :)
    algs = alg.algs isa IntegralAlgorithm ? ntuple(i -> alg.algs, Val(dim)) : alg.algs
    nest = init_nest(g, cacheval[dim][2], dom, p, lims, state, algs[1:dim-1], cacheval[1:dim-1]; kws...)
    return do_solve(nest, dom, (p, lims, state), algs[dim], cacheval[dim][1]; kws...)
end

"""
    AbsoluteEstimate(est_alg, abs_alg; kws...)

Most algorithms are efficient when using absolute error tolerances, but how do you know the
size of the integral? One option is to estimate it using second algorithm.

A multi-algorithm to estimate an integral using an `est_alg` to generate a rough estimate of
the integral that is combined with a user's relative tolerance to re-calculate the integral
to higher accuracy using the `abs_alg`. The keywords passed to the algorithm may include
`reltol`, `abstol` and `maxiters` and are given to the `est_alg` solver. They should limit
the amount of work of `est_alg` so as to only generate an order-of-magnitude estimate of the
integral. The tolerances passed to `abs_alg` are `abstol=max(abstol,reltol*norm(I))` and
`reltol=0`.
"""
struct AbsoluteEstimate{E<:IntegralAlgorithm,A<:IntegralAlgorithm,F,K<:NamedTuple} <: IntegralAlgorithm
    est_alg::E
    abs_alg::A
    norm::F
    kws::K
end
function AbsoluteEstimate(est_alg, abs_alg; norm=norm, kwargs...)
    kws = NamedTuple(kwargs)
    checkkwargs(kws)
    return AbsoluteEstimate(est_alg, abs_alg, norm, kws)
end

function init_cacheval(f, dom, p, alg::AbsoluteEstimate)
    return (est=init_cacheval(f, dom, p, alg.est_alg),
            abs=init_cacheval(f, dom, p, alg.abs_alg))
end

function do_solve(f, dom, p, alg::AbsoluteEstimate, cacheval;
                    abstol=nothing, reltol=nothing, maxiters=typemax(Int))
    sol = do_solve(f, dom, p, alg.est_alg, cacheval.est; alg.kws...)
    val = alg.norm(sol.u) # has same units as sol
    rtol = reltol === nothing ? sqrt(eps(one(val))) : reltol # use the precision of the solution to set the default relative tolerance
    atol = max(abstol === nothing ? zero(val) : abstol, rtol*val)
    return do_solve(f, dom, p, alg.abs_alg, cacheval.abs;
                    abstol=atol, reltol=zero(rtol), maxiters=maxiters)
end
