using Test
using LinearAlgebra

using StaticArrays
using OffsetArrays

using AutoBZCore


function integer_lattice(n)
    C = OffsetArray(zeros(ntuple(_ -> 3, n)), repeat([-1:1], n)...)
    for i in 1:n, j in (-1, 1)
        C[CartesianIndex(ntuple(k -> k == i ? j : 0, n))] = 1/2n
    end
    C
end

@testset "AutoBZCore" begin

    @testset "SymmetricBZ" begin
        dims = 3
        A = I(dims)
        B = AutoBZCore.canonical_reciprocal_basis(A)
        lims = AutoBZCore.CubicLimits(zeros(3), ones(3))
        fbz = FullBZ(A, B, lims)
        @test fbz == SymmetricBZ(A, B, lims, nothing)
        @test fbz.A ≈ A
        @test fbz.B ≈ B
        @test nsyms(fbz) == 1
        @test fbz.syms === nothing
        @test ndims(fbz) == dims
        @test eltype(fbz) == float(eltype(B))
        fbz = FullBZ(A)
        @test fbz.lims isa AutoBZCore.CubicLimits

        nsym = 8
        syms = rand(SMatrix{dims,dims}, nsym)
        bz = SymmetricBZ(A, B, lims, syms)
        @test nsyms(bz) == nsym
        @test bz.syms == syms
        @test ndims(bz) == dims
        @test eltype(bz) == float(eltype(B))
    end

    @testset "FourierIntegrand" begin
        dos_integrand(H::AbstractMatrix, M) = imag(tr(inv(M-H)))/(-pi)
        s = InplaceFourierSeries(rand(SMatrix{3,3,ComplexF64}, 3,3,3))
        p = -I
        f = AutoBZCore.construct_integrand(FourierIntegrand(dos_integrand, s), false, tuple(p))
        @test f == FourierIntegrand(dos_integrand, s, p)
        @test AutoBZCore.iterated_integrand(f, (1.0, 1.0, 1.0), Val(1)) == f((1.0, 1.0, 1.0))
        @test AutoBZCore.iterated_integrand(f, 809, Val(0)) == AutoBZCore.iterated_integrand(f, 809, Val(2)) == 809
    end

    dims = 3
    A = I(dims)
    B = AutoBZCore.canonical_reciprocal_basis(A)
    fbz = FullBZ(A, B)
    bz = SymmetricBZ(A, B, fbz.lims, (I,))

    dos_integrand(H, M) = imag(tr(inv(M-H)))/(-pi)      # test integrand with positional arguments
    p_dos = MixedParameters(complex(1.0,1.0)*I)
    
    gloc_integrand(h_k; η, ω) = inv(complex(ω,η)*I-h_k) # test integrand with keyword arguments
    p_gloc = MixedParameters(; η=1.0, ω=0.0)

    s = InplaceFourierSeries(integer_lattice(dims), period=1)

    for (integrand, p, T, args, kwargs...) in (
        (dos_integrand, p_dos, Float64, ([i*I for i in 1:3],), ),
        (gloc_integrand, p_gloc, ComplexF64, (), :η => ones(3), :ω => 1:3)
    )

        f = FourierIntegrand(integrand, s)

        ip_fbz = IntegralProblem(f, fbz, p)
        ip_bz = IntegralProblem(f, bz, p)

        @testset "IntegralProblem interface" begin
            g = FourierIntegrand(integrand, s, p)
            ip_fbz_g = IntegralProblem(g, fbz)
            ip_bz_g = IntegralProblem(g, bz)

            for (ip1, ip2) in ((ip_fbz, ip_fbz_g), (ip_bz, ip_bz_g))
                intf = AutoBZCore.construct_integrand(ip1.f, isinplace(ip1), ip1.p)
                intg = AutoBZCore.construct_integrand(ip2.f, isinplace(ip2), ip2.p)
                @test intf == intg
            end
        end

        @testset "Algorithms" begin
            @test solve(ip_fbz, IAI(); do_inf_transformation=Val(false)) ≈ solve(ip_bz, IAI(); do_inf_transformation=Val(false))
            @test solve(ip_fbz, TAI(); do_inf_transformation=Val(false)) ≈ solve(ip_bz, TAI(); do_inf_transformation=Val(false))
            @test solve(ip_fbz, PTR(); do_inf_transformation=Val(false)) ≈ solve(ip_bz, PTR(); do_inf_transformation=Val(false))
            @test solve(ip_fbz, AutoPTR(); do_inf_transformation=Val(false)) ≈ solve(ip_bz, AutoPTR(); do_inf_transformation=Val(false))
            @test solve(ip_fbz, PTR_IAI(); do_inf_transformation=Val(false)) ≈ solve(ip_bz, PTR_IAI(); do_inf_transformation=Val(false))
            @test solve(ip_fbz, AutoPTR_IAI(); do_inf_transformation=Val(false)) ≈ solve(ip_bz, AutoPTR_IAI(); do_inf_transformation=Val(false))
            # @test solve(ip_fbz, VEGAS()) ≈ solve(ip_bz, VEGAS()) # skip for now or
            # set larger tolerance
        end

        @testset "IntegralSolver" begin
            sol = IntegralSolver(f, fbz, IAI())
            @test sol(p.args...; p.kwargs...) == solve(ip_fbz, IAI(); do_inf_transformation=Val(false)).u
        end

        @testset "batchsolve" begin
            sol = IntegralSolver(f, fbz, IAI())
            @test [sol(p.args...; p.kwargs...) for p in AutoBZCore.paramzip(args, NamedTuple(kwargs))] ≈ batchsolve(T, sol, args...; kwargs...)
        end
    end

end