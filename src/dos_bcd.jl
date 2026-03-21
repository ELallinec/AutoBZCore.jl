using LinearAlgebra
using StaticArrays

function init_cacheval(h, domain, p, alg::BCD)
	h isa FourierSeries || throw(ArgumentError("BCD currently supports Fourier series Hamiltonians"))
	p isa SymmetricBZ || throw(ArgumentError("BCD supports BZ parameters from load_bz"))
	d2h = HessianSeries(h)
	wd2h = FourierSeriesEvaluators.workspace_allocate(d2h, FourierSeriesEvaluators.period(d2h))
	kalg = MonkhorstPack(npt = alg.npt, syms = p.syms)
	dom = canonical_ptr_basis(p.B)
	rule = init_fourier_rule(wd2h, dom, kalg)
	if rule isa FourierPTR
		if typeof(rule.s[1][1]) <: Number
			tabeigen = eigen.(getindex.(rule.s, 1))
		else
			tabeigen = eigen.(Hermitian.(getindex.(rule.s, 1)))
		end
	elseif rule isa FourierMonkhorstPack
		if typeof(rule.wxs[1][2].s) <: Number
			tabeigen = eigen.(getindex.(getproperty.(getindex.(rule.wxs, 2), :s), 1))
		else
			tabeigen = eigen.(Hermitian.(getindex.(getproperty.(getindex.(rule.wxs, 2), :s), 1)))
		end
	end
	return rule, tabeigen
end

function dos_solve(h, domain, p, alg::BCD, cacheval;
	abstol = 1e-8, reltol = nothing, maxiters = nothing)
	domain isa Number || throw(ArgumentError("BCD supports domains of individual eigenvalues"))
	E = domain
	bz = p
	rule, tabeigen = cacheval
	(; npt, α, ΔE, η) = alg
	d = ndims(bz)
	temp = rule isa FourierPTR ? rule.s[1][1] : rule.wxs[1][2].s[1]
	J = size(temp)[1]
	result = -imag(do_BCD(rule, tabeigen, h, E, α, ΔE, η, Val(d), Val(J)))
	return DOSSolution(result, Success, (;))
end



# Version optimisée de divided_difference_contour
function divided_difference_contour(sum, h, d1h, d2h, tabε, tabψ, g::AbstractMatrix{ComplexF64}, A::AbstractVector{Matrix{ComplexF64}}, ::Val{J}, ::Val{d}, E, ΔE, exp_mat) where {d, J}
	εs = (tabε .- E) ./ ΔE

	# Remplir g[n,m] = divided_difference_gaussian(εs[n], εs[m]) / ΔE (utilise buffer passé)
	@inbounds for n in 1:J
		en = εs[n]
		for m in 1:J
			g[n, m] = divided_difference_gaussian(en, εs[m]) / ΔE
		end
	end

	# Remplir A[i] = ψ' * d1h[i] * ψ dans les matrices préallouées A
	@inbounds for i in 1:d
		# Utilise deux mul! pour éviter des allocations intermédiaires lourdes
		temp = similar(A[i])
		mul!(temp, d1h[i], tabψ)
		mul!(A[i], tabψ', temp)
	end

	# Accumulation de Σ_{n,m} g[n,m] * A_i[n,m] * A_j[m,n]
	@inbounds for i in 1:d
		Ai = A[i]
		for j in 1:d
			Aj = A[j]
			s = 0.0 + 0.0im
			for n in 1:J
				for m in 1:J
					s += g[n, m] * Ai[n, m] * Aj[m, n]
				end
			end
			sum[i, j] += s
		end
	end

	# Contribution hessienne - utilisation de mul! pour éviter une allocation temporaire
	@inbounds for i in 1:d
		d2h_i = d2h[i]          # tuple des matrices pour la i-ème ligne de la hessienne
		for j in 1:(d+1-i)      # j indexe le décalage en colonne (j = colonne - i + 1)
			d2h_ij = d2h_i[j]
			temp = similar(d2h_ij)
			mul!(temp, d2h_ij, exp_mat)
			sum[i, j+i-1] += tr(temp)
		end
	end

	return Hermitian(sum)
end

function divided_difference_gaussian(x, y)
	if abs(x - y) <= 1e-16
		temp = -2x * exp(-x * x)
	elseif x == -y
		temp = 0.0
	elseif exp(-x^2) <= 1e-16 || exp(-y^2) <= 1e-16
		temp = (exp(-x^2) - exp(-y^2)) / (x - y)
	else
		temp = exp(-y * y) * (expm1((y - x) * (x + y)) / (x - y))
	end
	return temp
end


@inline function do_BCD(rule::FourierPTR, tabeigen, h, E, α, ΔE, η, ::Val{d}, ::Val{J}) where {d, J}
	DOS = 0.0 + 0.0im
	norm_factor = π * length(rule.p)
	# Préallocation des buffers g et A pour réutilisation sur chaque point k
	g = Matrix{ComplexF64}(undef, J, J)
	A = Vector{Matrix{ComplexF64}}(undef, d)
	@inbounds for i in 1:d
		A[i] = Matrix{ComplexF64}(undef, J, J)
	end
	@inbounds for index in CartesianIndices(rule.p)
		H, d1H, d2H = rule.s[index]
		k = rule.p[index][2]
		eig = tabeigen[index]
		DOS += bcd_integrand_eval(h, k, H, d1H, d2H, eig.values, eig.vectors, g, A, Val(d), Val(J), E, α, η, ΔE)
	end
	return DOS / norm_factor
end


@inline function do_BCD(rule::FourierMonkhorstPack, tabeigen, h, E, α, ΔE, η, ::Val{d}, ::Val{J}) where {d, J}
	DOS = 0.0 + 0.0im
	norm_factor = π * rule.npt^d
	# Préallocation des buffers g et A pour réutilisation sur chaque point k
	g = Matrix{ComplexF64}(undef, J, J)
	A = Vector{Matrix{ComplexF64}}(undef, d)
	@inbounds for i in 1:d
		A[i] = Matrix{ComplexF64}(undef, J, J)
	end
	@inbounds for index in CartesianIndices(rule.wxs)
		H, d1H, d2H = rule.wxs[index][2].s
		k = rule.wxs[index][2].x
		w = rule.wxs[index][1]
		eig = tabeigen[index]

		DOS += w * bcd_integrand_eval(h, k, H, d1H, d2H, eig.values, eig.vectors, g, A, Val(d), Val(J), E, α, η, ΔE)
	end
	return DOS / norm_factor
end


@inline function bcd_integrand_eval(h, k::SVector, H, d1H, d2H, evals, evecs, g::AbstractMatrix{ComplexF64}, A::AbstractVector{Matrix{ComplexF64}}, ::Val{d}, ::Val{J}, E, α, η, ΔE) where {d, J}
	M = evecs * Diagonal(exp.(-((evals .- E)/ΔE) .^ 2)) * evecs'
	def = SVector{d}(ntuple(i -> tr(d1H[i] * M), Val(d)))
	k_shifted::SVector{d, ComplexF64} = k - im * α * def
	ddef = SMatrix{d, d}(divided_difference_contour(zeros(ComplexF64, d, d), H, d1H, d2H, evals, evecs, g, A, Val(J), Val(d), E, ΔE, M))

	return tr(inv((E + im * η) * I - h(k_shifted))) * det(I - im * α * ddef)
end

