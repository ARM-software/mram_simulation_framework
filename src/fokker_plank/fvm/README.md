Notes regarding the implementation of Eq 3 in https://arxiv.org/pdf/2106.12304
(Read eq 3.1 to eq 3.9 in http://ieeexplore.ieee.org/document/6242414/, the golden truth:

Switching Distributions for Perpendicular Spin-Torque Devices Within the Macrospin Approximation)

# Implementing the Spherical Fokker–Planck (Eq. 3) in the Codebase

This note explains how the paper’s θ-form Fokker–Planck equation is implemented in the repository, and why the explicit metric factor $1/\sin\theta$ does **not** appear in the code.


## `fvm_classes.py` 
`fvm_classes.py` Finite Volume Method classes, see [FVM](https://github.com/danieljfarrell/FVM and [FVM2](https://github.com/Fratorhe/FVM/))

---

## 1) Equation in $ \theta $ (paper) and change of variables

Paper’s (axisymmetric) conservative form in polar angle $ \theta $:

$$
\frac{\partial \rho}{\partial t}
= -\frac{1}{\sin\theta}\frac{\partial}{\partial \theta}\Big[
\underbrace{\sin^2\theta(i-h-\cos\theta)\rho}_{\text{drift}}
-\underbrace{\frac{1}{2\Delta}\sin^2\theta\frac{\partial \rho}{\partial \theta}}_{\text{diffusion}}
\Big].
$$

Use the substitution $$z=\cos\theta$$ so that
$$\sin^2\theta = 1 - z^2$$ and $$\partial_\theta = -\sin\theta\,\partial_z$$.
With this change, the **metric factor disappears** and the PDE becomes a plain **z-divergence**:

$$
\frac{\partial \rho}{\partial t}
= -\frac{\partial}{\partial z}\Big[
\underbrace{(i-h-z)(1-z^2)\rho}_{U(z)\rho}
-\underbrace{\frac{1}{2\Delta}(1-z^2)\frac{\partial \rho}{\partial z}}_{D(z)\partial_z\rho}
\Big].
$$

Define the **flux**

$$
\[
J(z) = U(z)\rho - D(z)\partial_z\rho,
\qquad
U(z)=(i-h-z)(1-z^2),\quad
D(z)=\frac{1-z^2}{2\Delta},
\]
$$

then the PDE is $$\partial_t \rho = -\partial_z J$$.

---

## 2) How we encode it:

### Problem setup in **z** (MTJ): `src/fokker_plank/fvm/mtj_fp_fvm.py`

**Mesh in $z=\cos\theta$** (uniform in θ, non-uniform in z):
*$$sin^2(theta)$$ becomes $$(1-z^2)$$*
```
faces = np.cos(np.linspace(np.pi, 0, dim_points))  # z = cos θ faces
mesh = fvm.Mesh(faces)

U = (i0 - h - mesh.cells) * (1 - mesh.cells * mesh.cells)        # (i-h-z)(1-z^2)
D = (1 - mesh.cells * mesh.cells) / (2 * delta)                   # (1 - z^2)/(2Δ)
```
