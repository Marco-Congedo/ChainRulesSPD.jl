# Documentation

This package implements efficient **primal functions** on the *Riemannian Manifold of symmetric positive definite (SPD) matrices*
and all corresponding **backpropagation rules**, expanding those found in [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl).

These functions constitute the fundamental ingredients to build layers in SPD Riemannian Neuronal Networks.

---

## 🔗 Reference

Congedo M. Bouchard F. and Bhatia R. (2026) A new look at the Riemannian manifold of positive definite matrices,
*Transactions on Machine Learning research*.

---

## 🧩 Requirements 

**Julia**: version ≥ 1.13

---

## ⚙️ Installation

```julia
using Pkg
Pkg.add("ChainRulesSPD")
```

---

## —͟͟͞͞★ Quick Start
Read the [next page](@ref "ChainRulesSPD").

---
## 🛠️ TroubleShoothing

| Problem   | Solution |
|:----------|:----------|
| `softplus` collides with other packages | use: `import ChainRulesSPD: softplus`|

---

## ✍️ About the Author

[Marco Congedo](https://sites.google.com/site/marcocongedo), corresponding author, is a Research Director of [CNRS](http://www.cnrs.fr/en) (Centre National de la Recherche Scientifique), working at [UGA](https://www.univ-grenoble-alpes.fr/english/) (University of Grenoble Alpes). 

---

## 🧑‍💻 Contributors

---

## 🧪 Package Test

```julia
using Pkg
Pkg.test()
```
---

## 🧭 Index

```@index
```
