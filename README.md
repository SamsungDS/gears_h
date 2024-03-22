Everything here is more of a wishlist than a guarantee unless explicitly stated otherwise. Such is research.

# Surrogate LCAO Hamiltonians

SLH seeks to surrogatize and approximate density functional theory (or otherwise) hamiltonians and (optionally) overlap matrices in a basis of localized spherical orbitals.


# Architecture

Everything here is subject to rapid and aggressive changes without notice.

H = Hamiltonian matrix.\
S = Overlap matrix.\
T = Kinetic operator matrix.\
V = Potential operator matrix.

1. Onsite H blocks are connected to local atomic environments: The descriptors for the local atomic environments are equivariant with (optional) message-passing components.
2. Off-site H blocks are connected to EITHER a *pair* of local atomic environments from the two atoms that make up the bond, or a bond local environment. This is not decided yet.

3. Off-site S blocks are described using purely an equivariant expansion of the bond length and orientation. This is because the S matrix blocks are not environment dependent and are exact in the Slater-Koster limit.

**Potentially**:

1. We can train the environment dependence purely on the potential matrix elements which are the only environment-dependent part. S and T both get trained on the bonds (for the off-site case). I don't know how to deal with the onsite T case yet.