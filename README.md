# Surrogate LCAO Hamiltonians

SLH seeks to surrogatize and approximate density functional theory (or otherwise) hamiltonians and (optionally) overlap matrices in a basis of localized spherical orbitals.

# Architecture

H = Hamiltonian matrix.\
S = Overlap matrix.\
T = Kinetic operator matrix.\
V = Potential operator matrix.

1. Onsite H blocks are connected to local atomic environments: The descriptors for the local atomic environments are 2-body distance expansions with 1-2 layers of self-attention (message passing with softmax weights).

2. Off-site H blocks are connected to a pair of local atomic environments from the two atoms that make up the bond. The bond direction information is tensor producted with the sum of the atomic environment descriptors. In principle one can also tensor product the two atom-centered descriptors, but for errors down to <5e-3 eV for matrix irreps, this seems to not matter.

3. On- and off-site S matrix blocks are currently not implemented. They are relatively easy to calculate from GPAW and are purely 2-body objects with no environment dependence. However, they need to be quite accurate, so we will follow an alternative approach to approximating them.

**Potentially**:

1. We can train the environment dependence purely on the potential matrix elements which are the only environment-dependent part. S and T both get trained on the bonds (for the off-site case).
