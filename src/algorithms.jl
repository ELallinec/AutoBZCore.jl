abstract type AbstractAutoBZAlgorithm <: SciMLBase.AbstractIntegralAlgorithm end

struct IAI{F,I,S} <: AbstractAutoBZAlgorithm
    order::Int
    norm::F
    initdivs::I
    segbufs::S
end
IAI(; order=7, norm=norm, initdivs=nothing, segbufs=nothing) = IAI(order, norm, initdivs, segbufs)

struct PTR{R} <: AbstractAutoBZAlgorithm
    npt::Int
    rule::R
end
PTR(; npt=50, rule=nothing) = PTR(npt, rule)

struct AutoPTR{F,B} <: AbstractAutoBZAlgorithm
    norm::F
    buffer::B
end
AutoPTR(; norm=norm, buffer=nothing) = AutoPTR(norm, buffer)

struct PTR_IAI{P,I} <: AbstractAutoBZAlgorithm
    ptr::P
    iai::I
end
PTR_IAI(; ptr=PTR(), iai=IAI()) = PTR_IAI(ptr, iai)

struct AutoPTR_IAI{P,I} <: AbstractAutoBZAlgorithm
    reltol::Float64
    ptr::P
    iai::I
end
AutoPTR_IAI(; reltol=1.0, ptr=AutoPTR(), iai=IAI()) = AutoPTR_IAI(reltol, ptr, iai)

struct TAI{T<:HCubatureJL} <: AbstractAutoBZAlgorithm
    rule::T
end
TAI(; rule=HCubatureJL()) = TAI(rule)

# Imitate original interface
IntegralProblem(f, bz::SymmetricBZ, args...; kwargs...) =
    IntegralProblem{isinplace(f, 3)}(f, bz, bz, args...; kwargs...)

# layer to intercept integrand construction
function construct_integrand(f, iip, p)
    if iip
        (y, x) -> (f(y, x, p); y)
    else
        x -> f(x, p)
    end
end

function __solvebp_call(prob::IntegralProblem, alg::AbstractAutoBZAlgorithm,
                                sensealg, bz::SymmetricBZ, ::SymmetricBZ, p;
                                reltol = nothing, abstol = nothing, maxiters = typemax(Int))

    abstol_ = (abstol===nothing) ? zero(eltype(bz)) : abstol
    reltol_ = (abstol===nothing) ? (iszero(abstol_) ? sqrt(eps(typeof(abstol_))) : zero(abstol_)) : reltol
    f = construct_integrand(prob.f, isinplace(prob), prob.p)

    if alg isa IAI
        j = abs(det(bz.B))  # include jacobian determinant for map from fractional reciprocal lattice coordinates to Cartesian reciprocal lattice
        atol = abstol_/nsyms(bz)/j # reduce absolute tolerance by symmetry factor
        val, err = nested_quadgk(f, bz.lims; atol=atol, rtol=reltol_, maxevals = maxiters,
                                        norm = alg.norm, order = alg.order, initdivs = alg.initdivs, segbufs = alg.segbufs)
        val, err = symmetrize(f, bz, j*val, j*err)
        SciMLBase.build_solution(prob, alg, val, err, retcode = ReturnCode.Success)
    elseif alg isa PTR
        val = symptr(f, bz.B, bz.syms; npt = alg.npt, rule = alg.rule)
        val = symmetrize(f, bz, val)
        err = nothing
        SciMLBase.build_solution(prob, alg, val, err, retcode = ReturnCode.Success)
    elseif alg isa AutoPTR
        val, err = autosymptr(f, bz.B, bz.syms;
                        atol = abstol_, rtol = reltol_, maxevals = maxiters, norm=alg.norm, buffer=alg.buffer)
        val, err = symmetrize(f, bz, val, err)
        SciMLBase.build_solution(prob, alg, val, err, retcode = ReturnCode.Success)
    elseif alg isa PTR_IAI
        sol = __solvebp_call(prob, alg.ptr, sensealg, bz, bz, p;
                                reltol = reltol_, abstol = abstol_, maxiters = maxiters)
        atol = max(abstol_, reltol_*alg.iai.norm(sol))
        __solvebp_call(prob, alg.iai, sensealg, bz, bz, p;
                                reltol = zero(atol), abstol = atol, maxiters = maxiters)
    elseif alg isa AutoPTR_IAI
        sol = __solvebp_call(prob, alg.ptr, sensealg, bz, bz, p;
                                reltol = alg.reltol, abstol = abstol_, maxiters = maxiters)
        atol = max(abstol_, reltol_*alg.iai.norm(sol))
        __solvebp_call(prob, alg.iai, sensealg, bz, bz, p;
                                reltol = zero(atol), abstol = atol, maxiters = maxiters)
    elseif alg isa TAI
        l = lattice_bz_limits(bz.B); a = l.a; b = l.b
        j = abs(det(bz.B))
        sol = __solvebp_call(prob, alg.rule, sensealg, a, b, p;
                                abstol=abstol_, reltol=reltol_, maxiters=maxiters)
        SciMLBase.build_solution(sol.prob, sol.alg, sol.u*j, sol.resid*j, retcode = sol.retcode, chi = sol.chi)
    end
end

