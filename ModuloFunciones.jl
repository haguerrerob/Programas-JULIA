module ModuloFunciones
using LinearAlgebra, LaTeXStrings, Symbolics

@variables r s
    Dr = Differential(r)
    Ds = Differential(s)

function matrizConectividadGlobal(numNodosx, numNodosy, numElementos)

    matrizConectividadG = zeros(Int, numElementos, 4)
    elemento = 1

    for j in 1:numNodosy-1
        for i in 1:numNodosx-1
            nodoInferiorIzquierdo::Int = i + ((j - 1) * numNodosx)
            matrizConectividadG[elemento, :] = [(nodoInferiorIzquierdo + numNodosx + 1) (nodoInferiorIzquierdo + numNodosx) nodoInferiorIzquierdo (nodoInferiorIzquierdo + 1)]
            elemento += 1
        end
    end

    return matrizConectividadG
end

function mallador(numNodosx, numNodosy, longx, longy, numElementos)

    numSegmentosx = numNodosx - 1
    numSegmentosy = numNodosy - 1
    dNodosx = longx / numSegmentosx
    dNodosy = longy / numSegmentosy 
    #numNodosG = numNodosx * numNodosy
    matrizCoordG = zeros(numNodosx * numNodosy, 2)
    Coordy = 0
    nodo = 1

    for j in 1:numNodosy
        Coordx = 0
        for i in 1:numNodosx
            matrizCoordG[nodo, :] = [Coordx Coordy]
            Coordx += dNodosx
            nodo += 1
        end
        Coordy += dNodosy
    end

    return matrizCoordG
end

function matrizGdGL(numNodosx, numNodosy, mConectividadGlobal)

    numElementos = (numNodosx - 1) * (numNodosy - 1)
    numNodosGlobales = numNodosx * numNodosy
    mGdGL = zeros(Integer, 8, numElementos)

    for i in 1:numElementos
        j = 1
        while j <= 4
            zz = mConectividadGlobal[i, :]
            mGdGL[2*j-1, i] = zz[j] * 2 - 1
            mGdGL[2*j, i] = zz[j] * 2
            j += 1
        end
    end

    return mGdGL
end

function Jacobiano(∇H, mGCoorNodales, mConectG, el)

    # mNodalCoords --> matriz de coordenadas nodales del elemento el.
    mNodalCoords₀ = zeros(4, 2)
    for nodo in 1:4
        mNodalCoords₀[nodo, :] = mGCoorNodales[mConectG[el, :][nodo], :]
    end

    #x₀ = expand((H * mNodalCoords[:, 1])[1])
    #y₀ = expand((H * mNodalCoords[:, 2])[1])

    # J - Jacobiano. Fundamentals of FEA, Koutromanos-pag.240 (ec.8.5.25)
    J = expand.(∇H * mNodalCoords₀)

    #=
    dxdr = expand_derivatives(Dr(x₀))
    dxds = expand_derivatives(Ds(x₀))
    dydr = expand_derivatives(Dr(y₀))
    dyds = expand_derivatives(Ds(y₀))

    J = [dxdr dydr; dxds dyds] # Jacobiano
    =#

    return J
end

function BL0(∇H, J)
 
    B = inv(J) * ∇H

    BL0₁ = [B[1, 1] 0 B[1,2] 0 B[1,3] 0 B[1,4] 0]
    BL0₂ = [0 B[2,1] 0 B[2, 2] 0 B[2,3] 0 B[2,4]]
    BL0₃ = [B[2,1] B[1, 1] B[2, 2] B[1,2] B[2,3] B[1,3] B[2,4] B[1,4]]

    BL0 = vcat(BL0₁, BL0₂, BL0₃)

    return BL0
end

function BL1(∇H, J, Uₑ)
    
    B = inv(J) * ∇H

    l₁₁, l₁₂, l₂₁, l₂₂ = 0, 0, 0, 0

    for k in 1:4 l₁₁ += B[1, k] * Uₑ[k, 1] end
    for k in 1:4 l₁₂ += B[2, k] * Uₑ[k, 1] end
    for k in 1:4 l₂₁ += B[1, k] * Uₑ[k, 2] end
    for k in 1:4 l₂₂ += B[2, k] * Uₑ[k, 2] end

    BL1₁ = [l₁₁ * B[1, 1] l₂₁ * B[1, 1] l₁₁ * B[1, 2] l₂₁ * B[1, 2] l₁₁ * B[1, 3] l₂₁ * B[1, 3] l₁₁ * B[1, 4] l₂₁ * B[1, 4]]
    BL1₂ = [l₁₂ * B[2, 1] l₂₂ * B[2, 1] l₁₂ * B[2, 2] l₂₂ * B[2, 2] l₁₂ * B[2, 3] l₂₂ * B[2, 3] l₁₂ * B[2, 4] l₂₂ * B[2, 4]]
    BL1₃ = [(l₁₁ * B[2, 1] + l₁₂ * B[1, 1]) (l₂₁ * B[2, 1] + l₂₂ * B[1, 1]) (l₁₁ * B[2, 2] + l₁₂ * B[1, 2]) (l₂₁ * B[2, 2] + l₂₂ * B[1, 2]) (l₁₁ * B[2, 3] + l₁₂ * B[1, 3]) (l₂₁ * B[2, 3] + l₂₂ * B[1, 3]) (l₁₁ * B[2, 4] + l₁₂ * B[1, 4]) (l₂₁ * B[2, 4] + l₂₂ * B[1, 4])]
    
    BL1 = vcat(BL1₁, BL1₂, BL1₃)

    return BL1
end

# TGD -> Tensor Gradiente de Deformaciones
function TGD(Uₑ, J, mGCoorNodales, mConectG, el, H)
    mNodalCoordsₜ = zeros(4, 2)
    for nodo in 1:4
        mNodalCoordsₜ[nodo, :] = mGCoorNodales[mConectG[el, :][nodo], :] + Uₑ[nodo, :]
    end

    xₜ = expand((H * mNodalCoordsₜ[:, 1])[1])
    yₜ = expand((H * mNodalCoordsₜ[:, 2])[1])

    dxₜdr = expand_derivatives(Dr(xₜ))
    dxₜds = expand_derivatives(Ds(xₜ))
    dyₜdr = expand_derivatives(Dr(yₜ))
    dyₜds = expand_derivatives(Ds(yₜ))

    X₁ = inv(J)
    X₂ = [dxₜdr dyₜdr; dxₜds dyₜds]

    X = X₂ * X₁ # X -> TENSOR GRADIENTE DE DEFORMACIONES
    return X

end

# Matriz de Rigidez Global Lineal SOLO PARA LA PRIMERA ITERACION *******
function KGL00(∇H, mConectG, mGCoorNodales, C, numElementos, GDLG, mGdlG, GL9Q, wGL9Q)

    KGL00 = zeros(Float64, GDLG, GDLG)

    for el in 1:numElementos
        KEL0 = zeros(Float64, 8, 8)
        GLE = mGdlG[:, el]
        
        J = Jacobiano(∇H, mGCoorNodales, mConectG, el)
        BL₀ = BL0(∇H, J)
        
        for i in 1:9 # Integración en los puntos de Gauss
            Jg_det = substitute(det(J), Dict([r => GL9Q[i, 1], s => GL9Q[i, 2]]))
            Bg = substitute(BL₀, Dict([r => GL9Q[i, 1], s => GL9Q[i, 2]]))
            KEL0 += (transpose(Bg) * C * Bg * Jg_det * wGL9Q[i, 1] * wGL9Q[i, 2])
        end

        for i in eachindex(GLE)
            for j in eachindex(GLE)
                KGL00[GLE[i], GLE[j]] += KEL0[i, j].val
            end
        end
    end
    return KGL00
    
end

function KEL0(el, ∇H, mConectG, mGCoorNodales, C, GL9Q, wGL9Q)

    KEL0 = zeros(Float64, 8, 8)
        
    J = Jacobiano(∇H, mGCoorNodales, mConectG, el)
    BL₀ = BL0(∇H, J)
        
    for i in 1:9 # Integración en los puntos de Gauss
        Jg_det = substitute(det(J), Dict([r => GL9Q[i, 1], s => GL9Q[i, 2]]))
        Bg = substitute(BL₀, Dict([r => GL9Q[i, 1], s => GL9Q[i, 2]]))
        KEL0 += (transpose(Bg) * C * Bg * Jg_det * wGL9Q[i, 1] * wGL9Q[i, 2])
    end

    return KEL0
    
end
#=
function KGL1(∇H, mConectG, mGCoorNodales, C, numElementos, GDLG, mGdlG, GL9Q, wGL9Q, U)
    KGL = zeros(Float64, GDLG, GDLG)
    for el in 1:numElementos
        KEL = zeros(Float64, 8, 8)
        mGdlE = mGdlG[:, el]
        numGLE = length(mGdlE)
        Uₑ = transpose(reshape(U[mGdlE], 2, 4))
        
        J = Jacobiano(∇H, mGCoorNodales, mConectG, el)
        BL₀ = BL0(∇H, J)
        BL₁ = BL1(∇H, J, Uₑ)
        BL = BL₀ + BL₁
        
        for i in 1:9 # Integración en los puntos de Gauss

            Jg_det = substitute(det(J), Dict([r => GL9Q[i, 1], s => GL9Q[i, 2]]))
            Bg = substitute(BL, Dict([r => GL9Q[i, 1], s => GL9Q[i, 2]]))

            KEL += (transpose(Bg) * C * Bg * Jg_det * wGL9Q[i, 1] * wGL9Q[i, 2])

        end

        for i in 1:numGLE
            for j in 1:numGLE
                KGL[mGdlE[i], mGdlE[j]] += KEL[i, j].val
            end
        end
    end
    return KGL
end
=#
function KEL1(el, ∇H, mConectG, mGCoorNodales, C, mGdlG, GL9Q, wGL9Q, U)

    KEL1 = zeros(Float64, 8, 8)
    mGdlE = mGdlG[:, el]
    numGLE = length(mGdlE)
    Uₑ = transpose(reshape(U[mGdlE], 2, 4))
       
    J = Jacobiano(∇H, mGCoorNodales, mConectG, el)
    BL₀ = BL0(∇H, J)
    BL₁ = BL1(∇H, J, Uₑ)
    BL = BL₀ + BL₁
        
    for i in 1:9 # Integración en los puntos de Gauss
        Jg_det = substitute(det(J), Dict([r => GL9Q[i, 1], s => GL9Q[i, 2]]))
        Bg = substitute(BL, Dict([r => GL9Q[i, 1], s => GL9Q[i, 2]]))
        KEL += (transpose(Bg) * C * Bg * Jg_det * wGL9Q[i, 1] * wGL9Q[i, 2])
    end
    return KEL1
end
#=
function KGNL(mConectG, mGCoorNodales, C, numElementos, GDLG, mGdlG, GL9Q, wGL9Q, U, H, ∇H)
    KGNL = zeros(Float64, GDLG, GDLG)
    for el in 1:numElementos
        mGdlE = mGdlG[:, el]
        numGLE = length(mGdlE)
        KENL = zeros(numGLE, numGLE)
        Uₑ = transpose(reshape(U[mGdlE], 2, 4))

        J = Jacobiano(∇H, mGCoorNodales, mConectG, el)
        B = inv(J) * ∇H
        
        X = TGD(Uₑ, J, mGCoorNodales, mConectG, el, H) # TENSOR GRADIENTE DE DEFORMACIONES    
        ϵ = 1 / 2 * (transpose(X) * X - Matrix(I, 2, 2)) # TENSOR DE DEFORMACIONES DE GREEN-LAGRANGE
        ϵv = [ϵ[1, 1]; ϵ[2, 2]; ϵ[1, 2]] # VECTOR DE DEFORMACIONES DE GREEN-LAGRANGE
        S = C * ϵv # S -> SEGUNDO TENSOR DE PIOLA KIRCHHOFF
        ST = [S[1,1] S[3,1] 0 0; S[3,1] S[2,1] 0 0; 0 0 S[1,1] S[3,1]; 0 0 S[3,1] S[2,1]]

        BNL1 = [B[1,1] 0 B[1,2] 0 B[1,3] 0 B[1,4] 0]
        BNL2 = [B[2,1] 0 B[2,2] 0 B[2,3] 0 B[2,4] 0]
        BNL3 = [0 B[1,1] 0 B[1,2] 0 B[1,3] 0 B[1,4]]
        BNL4 = [0 B[2,1] 0 B[2,2] 0 B[2,3] 0 B[2,4]]
        BNL = vcat(BNL1, BNL2, BNL3, BNL4)

        for i in 1:9 # Integración en los puntos de Gauss
            BNLG = substitute(BNL, Dict([r => GL9Q[i, 1], s => GL9Q[i, 2]]))
            STG = substitute(ST, Dict([r => GL9Q[i, 1], s => GL9Q[i, 2]]))
            KENL += (transpose(BNLG) * STG * BNLG * wGL9Q[i, 1] * wGL9Q[i, 2])
        end

        for i in 1:numGLE
            for j in 1:numGLE
                KGNL[mGdlE[i], mGdlE[j]] += KENL[i, j]
            end
        end
        
    end

    return KGNL
    
end
=#
function KENL(el, mConectG, mGCoorNodales, C, mGdlG, GL9Q, wGL9Q, U, H, ∇H)
    mGdlE = mGdlG[:, el]
    numGLE = length(mGdlE)
    KENL = zeros(numGLE, numGLE)
    Uₑ = transpose(reshape(U[mGdlE], 2, 4))
    J = Jacobiano(∇H, mGCoorNodales, mConectG, el)
    B = inv(J) * ∇H
       
    X = TGD(Uₑ, J, mGCoorNodales, mConectG, el, H) # TENSOR GRADIENTE DE DEFORMACIONES    
    ϵ = 1 / 2 * (transpose(X) * X - Matrix(I, 2, 2)) # TENSOR DE DEFORMACIONES DE GREEN-LAGRANGE
    ϵv = [ϵ[1, 1]; ϵ[2, 2]; ϵ[1, 2]] # VECTOR DE DEFORMACIONES DE GREEN-LAGRANGE
    S = C * ϵv # S -> SEGUNDO TENSOR DE PIOLA KIRCHHOFF
    ST = [S[1,1] S[3,1] 0 0; S[3,1] S[2,1] 0 0; 0 0 S[1,1] S[3,1]; 0 0 S[3,1] S[2,1]]

    BNL1 = [B[1,1] 0 B[1,2] 0 B[1,3] 0 B[1,4] 0]
    BNL2 = [B[2,1] 0 B[2,2] 0 B[2,3] 0 B[2,4] 0]
    BNL3 = [0 B[1,1] 0 B[1,2] 0 B[1,3] 0 B[1,4]]
    BNL4 = [0 B[2,1] 0 B[2,2] 0 B[2,3] 0 B[2,4]]
    BNL = vcat(BNL1, BNL2, BNL3, BNL4)

    for i in 1:9 # Integración en los puntos de Gauss
        BNLG = substitute(BNL, Dict([r => GL9Q[i, 1], s => GL9Q[i, 2]]))
        STG = substitute(ST, Dict([r => GL9Q[i, 1], s => GL9Q[i, 2]]))
        KENL += (transpose(BNLG) * STG * BNLG * wGL9Q[i, 1] * wGL9Q[i, 2])
    end

    return KENL
    
end

function FGA(U, GDLG, H, ∇H, numElementos, mGdlG, mGCoorNodales, mConectG, C, GL9Q, wGL9Q)
    FG = zeros(Float64, GDLG)

    for el in 1:numElementos
        mGdlE = mGdlG[:, el]
        
        Uₑ = transpose(reshape(U[mGdlE], 2, 4))
        Fₑ = zeros(8, 1)

        J = Jacobiano(∇H, mGCoorNodales, mConectG, el)

        BL₀ = BL0(∇H, J)
        BL₁ = BL1(∇H, J, Uₑ)
        BL = BL₀ + BL₁

        X = TGD(Uₑ, J, mGCoorNodales, mConectG, el, H) # TENSOR GRADIENTE DE DEFORMACIONES
    
        ϵ = 1 / 2 * (transpose(X) * X - Matrix(I, 2, 2)) # TENSOR DE DEFORMACIONES DE GREEN-LAGRANGE

        ϵv = [ϵ[1, 1]; ϵ[2, 2]; ϵ[1, 2]] # VECTOR DE DEFORMACIONES DE GREEN-LAGRANGE

        S = C * ϵv # S -> SEGUNDO TENSOR DE PIOLA KIRCHHOFF

        for i in 1:9
            Sₑₗ = substitute(S, Dict([r => GL9Q[i, 1], s => GL9Q[i, 2]]))
            BLₑ = substitute(BL, Dict([r => GL9Q[i, 1], s => GL9Q[i, 2]]))
            Fₑ += transpose(BLₑ) * Sₑₗ * wGL9Q[i, 1] * wGL9Q[i, 2]
        end

        for i in eachindex(mGdlE) 
            FG[mGdlE[i]] += Fₑ[i] 
        end
    end
    return FG
end
    
end

function KGLOBAL(numElementos, mGdlG, U, ∇H, mGCoorNodales, mConectG, C, GL9Q, wGL9Q)
    KGLOBAL = zeros(Float64, GDLG, GDLG)

    for el in 1:numElementos
        mGdlE = mGdlG[:, el]
        numGLE = length(mGdlE)

        KEL₀ = KEL0(el, ∇H, mConectG, mGCoorNodales, C, GL9Q, wGL9Q)
        KEL₁ = KEL1(el, ∇H, mConectG, mGCoorNodales, C, mGdlG, GL9Q, wGL9Q, U)
        KENL = KENL(el, mConectG, mGCoorNodales, C, mGdlG, GL9Q, wGL9Q, U, H, ∇H)

        for i in 1:numGLE
            for j in 1:numGLE
                KGLOBAL[mGdlE[i], mGdlE[j]] += KEL₀[i, j] + KEL₁[i, j] + KENL[i, j]
            end
        end
    end

    return KGLOBAL

end
