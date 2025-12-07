Wavepacket Dynamics & Coulomb Scattering in Monolayer Graphene
A Real-Time Tight-Binding Simulation Framework

This repository contains a high-performance, real-time tight-binding simulation framework for modeling electron wavepacket propagation and Coulomb scattering in monolayer graphene. It implements a full time-dependent evolution of the quantum state on a graphene lattice, enabling direct visualization of:

- Wavepacket propagation
- Electron–hole scattering asymmetry
- Fano resonances & quasi-Rydberg states (supercritical regime)
- Transport cross sections from angle-resolved scattering
- Driven Coulomb impurities (adiabatic & non-adiabatic regimes)

The methods are based on the dissertation:
“Wavepacket Dynamics and Coulomb Scattering in Monolayer Graphene: A Tight-Binding Approach”.
All wavepacket propagation, Coulomb driving, transport analysis, and adaptive-domain optimizations were developed by the author.

FEATURES
✔ Full Time-Dependent Tight-Binding Evolution
✔ Monolayer Graphene Tight-Binding Model
✔ Gaussian Dirac Wavepacket Initialization
✔ Adaptive Active-Domain Propagation (Original Contribution)
✔ Perfectly Matched Layers (PML)
✔ Flux-Based Transmission/Reflection Calculation
✔ Transport Cross Section (New Method)
✔ Coulomb Impurity Physics (Attractive/Repulsive)
✔ Reproduction of V-shaped electron–hole mobility asymmetry
✔ Supercritical Coulomb regime: Fano resonances & quasi-Rydberg states
✔ Driven Coulomb impurities (adiabatic & non-adiabatic)

INSTALLATION
Dependencies:
- Python >= 3.9
- pybinding
- numpy / scipy
- matplotlib
- cupy (optional, for GPU acceleration)

Clone repository:
git clone https://github.com/sammcsherry/individual_project
cd individual_project

USAGE
1. Build graphene lattice
   lat = build_graphene(size_nm=1000)

2. Initialize wavepacket
   psi0 = make_wavepacket(lat, sigma=50, kx=0.25, ky=0)

3. Add Coulomb impurity
   H = add_coulomb(lat, alpha=-0.4)

4. Run time evolution
   psi_t = evolve(psi0, H, dt=1e-15, steps=500)

5. Extract transport cross section
   sigma_tr = compute_transport_cross_section(psi_t)

CITATION
Wavepacket Dynamics and Coulomb Scattering in Monolayer Graphene:
A Tight-Binding Approach — S. McSherry (2025)

