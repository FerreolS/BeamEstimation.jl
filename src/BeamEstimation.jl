module BeamEstimation

using  NFFT, Unitful, AbstractFFTs, Requires, LinearAlgebra
import UnitfulAngles: mas

export 	uvplane,
		dirtybeam,
		maxpixsize,
		maxfov,
		fastbeam,
		mas


maxpixsize_(λmin,Bmax) = λmin/Bmax/2 
maxpixsize(λmin::Real,Bmax::Real) = maxpixsize_(λmin,Bmax) 
maxpixsize(λmin::Unitful.Length,Bmax::Unitful.Length)  = (maxpixsize_(λmin,Bmax)  |> NoUnits) *1u"rad" |> mas
maxpixsize(λ::AbstractArray,Bmax) = maxpixsize(minimum(λ),Bmax)
maxpixsize(λ,baselines::AbstractMatrix) = maxpixsize(λ,maximum(sqrt.(sum(abs2,baselines,dims=2))))
maxpixsize(λ::AbstractArray,baselinesU::AbstractVector,baselinesV::AbstractVector) = maxpixsize(λ,maximum(sqrt.(abs2.(baselinesU) .+ abs2.(baselinesV))))

maxpixsize(uv::AbstractArray{Real,N}) where N = minimum((sum(abs2,uv[1],dims=1)).^(-1/2)) |> mas
maxpixsize(uv::AbstractArray{T,N}) where {N,T<:Unitful.Quantity} = minimum((sum(abs2,uv,dims=1)).^(-1/2)) / 2 |> mas

maxfov_(λmax,D) = λmax/D
maxfov(λmax::Real,D::Real) = maxfov_(λmax,D) 
maxfov(λmax::Unitful.Length,D::Unitful.Length) = (maxfov_(λmax,D)  |> NoUnits) *1u"rad" |> mas
maxfov(λ::AbstractArray,D) = maxfov(minimum(λ),D)

uvplane_(Baselines, λ) = Baselines ./ reshape(λ, 1,1,:)
function uvplane_(::Type{T},Baselines, λ) where T<:Unitful.Length
    uvplane_(Baselines, λ)  .|> u"rad^-1"
end
function uvplane_(::Type{T},Baselines, λ) where T<:Real
    uvplane_(Baselines, λ)  
end
uvplane(Baselines, λ) = uvplane_(eltype(Baselines),Baselines,λ)
uvplane(Baselines, λ::Number)=uvplane(Baselines, [λ])

function uvplane(BaselinesU,BaselinesV, λ)
	Baselines = vcat(reshape(BaselinesU, 1,:),reshape(BaselinesV, 1,:))
	uvplane(Baselines, λ)
end

function dirtybeam(uv,fov, pixsize)
	N = round(Int,fov/pixsize)
	scaleduv = hcat([0; 0], reshape(uv* pixsize,2,:))  .|> NoUnits
	nfftplan  = plan_nfft(scaleduv,(N,N));
	δ = fill(1.0 /(N^2) + 0im,size(scaleduv,2))
	δ[1,1] = 1/(N^2) 
	beam = nfftplan' * δ
	return real.(beam)
end



function fastbeam(uvp::AbstractArray{T,3}) where {T}
	N = length(uvp[1,:,:])
	if T<:Quantity
		uvp = ustrip.(u"rad^-1",uvp)
	end
	S11 = 2 * sum(uvp[1,:,:].^2 ) / (2N+1)
	S22 = 2 * sum(uvp[2,:,:].^2 ) / (2N+1)
	S12 = 2 * sum(uvp[1,:,:] .* uvp[2,:,:] ) / (2N+1)
	#S = SHermitianCompact{2,Float64}([ S11 S12; S12 S22])
	S = [ S11 S12; S12 S22]
	vals, vecs = eigen(S)
	rx,ry = sqrt(2*log(2))/(2*π) ./ sqrt.(vals) .* u"rad" .|> mas
	θ  = atan(u"rad",vecs[1,1],vecs[2,1])
	return rx,ry, θ
end



function __init__()
    @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" begin
		#using Plots
		function plotimage(image, fov;  kw...)
			sz = size(image)
			Plots.heatmap(fftshift(fftfreq(sz[1])).*fov,fftshift(fftfreq(sz[1])).*fov,image , ticks=:native,aspect_ratio=:equal; kw...)
		end

		function plotuv(uv; kw...)
			Plots.plot(vcat(uv[1,:,:],-uv[1,:,:]),vcat(uv[2,:,:],-uv[2,:,:]), seriestype=:scatter,ticks=:native,aspect_ratio=:equal; kw...)
		end

		plotbeam!(p,rx,ry,θ; kw...) = Plots.plot!(p,t-> rx * cos(θ) * cos(t)  - ry * sin(θ) * sin(t), t-> rx * sin(θ) * cos(t)  + ry * cos(θ) * sin(t), 0, 2π, line=4,leg=false,ticks=:native; kw...)
			
    end
end


function BuildCovariance(rx,ry,θ)
	Vx,Vy = ((rx,ry) ./(2*sqrt(2*log(2)))).^2
	R =  [ 	cos(θ)  -sin(θ) ;
			sin(θ)  cos(θ) ]
	S =  [ 	Vx  0  ;
			0  	Vy ]
	return R'*S*R
end

function Gaussian2D(tx,ty,W)
	r = tx.^2 * W[1,1] .- 2*W[1,2] *tx*ty' .+ (ty'.^2)* W[2,2]
	return 1/(2π) * sqrt(det(W)) .* exp.(- 0.5 .* r)
end

end # module BeamEstimation
