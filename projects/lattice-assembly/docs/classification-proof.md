# Finite saturation of rotation-closed three-patch families

This working proof accompanies the exact witness checker. Development details from the October 2026 manuscript revision are recorded in [provenance](../PROVENANCE.md). It has not been independently refereed; literature priority and globally minimum witness sizes are not claimed.

The theorem concerns finite connected bonded components on one triangular lattice. It does not concern wrapping components on a finite torus.

Index the nearest-neighbour directions of the triangular lattice in axial coordinates by
\[
d_0=(1,0),\quad d_1=(0,1),\quad d_2=(-1,1),\quad
d_3=(-1,0),\quad d_4=(0,-1),\quad d_5=(1,-1).
\]
Here \(d_{j+3}=-d_j\), with indices modulo six. A patch pattern is a three-element subset of these directions. Write \(\operatorname{rot}(P)\) for all six rotations of \(P\), allowing repetitions to collapse. The twenty patterns form four rotation orbits:
\[
A=\operatorname{rot}\{0,1,2\},\quad B=\operatorname{rot}\{0,1,3\},\quad
C=\operatorname{rot}\{0,1,4\},\quad D=\operatorname{rot}\{0,2,4\},
\]
of sizes \(6,6,6,2\). In particular, \(A\) consists of three consecutive directions and \(D\) of alternating directions. Orbits \(B\) and \(C\) are exchanged by reflection; they are distinct under rotation alone.

**Theorem (finite saturation for rotation-closed families).** Let \(\mathcal F\) be a nonempty subset of \(\{A,B,C,D\}\), and admit precisely the patch patterns in its union. A finite connected, fully bonded component exists if and only if \(A\in\mathcal F\) and \(\mathcal F\ne\{A\}\). Thus seven of the fifteen rotation-closed families admit finite saturation.

**Proof.** First, every finite saturated component contains an \(A\) particle. Choose a linear functional which is nonzero on all six neighbour directions and has a unique maximizer \(v\) on the component. All three bonded directions at \(v\) have strictly negative functional value. The three negative directions are consecutive, so the pattern at \(v\) belongs to \(A\).

We next exclude \(A\) alone. Suppose a finite saturated \(A\)-only component exists. Translate its highest axial row to \(y=0\), placing the rightmost vertex in that row at \(v_0=(0,0)\). A top-row particle has neither direction 1 nor direction 2, so its pattern is either
\[
L=\{3,4,5\}\quad\hbox{or}\quad R=\{4,5,0\}.
\]
The missing right neighbour forces \(v_0\) to have \(L\); reciprocity then forces \(w_0=(-1,0)\) to have \(R\). The particle at \(u=(1,-1)\), required by direction 5 of \(v_0\), contains direction 2 but cannot contain direction 1, since \((1,0)\) is absent from the component. Its pattern is therefore \(\{2,3,4\}\). Consequently \(m_0=(0,-1)\) receives directions 0,1,2 from \(u,v_0,w_0\), respectively, and has pattern \(M=\{0,1,2\}\).

For \(k\ge0\), put
\[
v_k=(-2k,0),\quad w_k=(-2k-1,0),\quad
m_k=(-2k,-1),\quad t_k=(-2k-1,-1).
\]
Assume \(v_k,w_k,m_k\) have patterns \(L,R,M\). Direction 4 of \(w_k\) requires \(t_k\), which contains direction 1. It cannot contain direction 0 because \(m_k\) lacks its reciprocal direction 3. The only \(A\) pattern containing 1 but excluding 0 is \(\{1,2,3\}\). Its direction 2 requires \(v_{k+1}\). This top-row particle cannot point right toward \(w_k\), which lacks direction 3; hence \(v_{k+1}\) has \(L\), and its left neighbour \(w_{k+1}\) has \(R\). Finally \(m_{k+1}\) receives directions 0,1,2 from \(t_k,v_{k+1},w_{k+1}\), so has \(M\). Induction forces infinitely many distinct top-row vertices, contradicting finiteness.

The appendix gives explicit reciprocal degree-three connected components for \(A\cup B\), \(A\cup C\), and \(A\cup D\), with respectively 12,12,16 particles. Adding allowed patterns preserves these witnesses. Together with the two exclusions, this proves the criterion. \(\square\)

The proof refers to the sites of a bonded component. Other particles may be adjacent through absent bonds; they do not change its highest row or supply a bonded neighbour without becoming part of the component. No assumption that the component is an induced lattice graph is used.

## Explicit saturated components

For each row below, place a particle at the stated axial coordinate with the listed three directions. A direction \(j\) from \((x,y)\) bonds to \((x,y)+d_j\); the endpoint carries direction \(j+3\) modulo six. These lists fully specify the physical bonds. Every listed patch is reciprocal, all degrees equal three, and each graph is connected. Particle/bond counts are 12/18 for \(AB\), 12/18 for \(AC\), and 16/24 for \(AD\). The lists prove existence without claiming global minimum size.

### AB witness

| Site | Axial coordinate | Patch directions |
|---:|:---:|:---:|
| 1 | (-1,-1) | {0,1,5} |
| 2 | (-1,0) | {1,4,5} |
| 3 | (-1,1) | {0,4,5} |
| 4 | (0,-2) | {0,1,2} |
| 5 | (0,-1) | {2,3,4} |
| 6 | (0,0) | {0,1,2} |
| 7 | (0,1) | {3,4,5} |
| 8 | (1,-2) | {0,1,3} |
| 9 | (1,-1) | {0,4,5} |
| 10 | (1,0) | {2,3,5} |
| 11 | (2,-2) | {1,2,3} |
| 12 | (2,-1) | {2,3,4} |

### AC witness

| Site | Axial coordinate | Patch directions |
|---:|:---:|:---:|
| 1 | (0,-2) | {0,1,5} |
| 2 | (0,-1) | {0,1,4} |
| 3 | (0,0) | {0,4,5} |
| 4 | (1,-3) | {0,1,2} |
| 5 | (1,-2) | {3,4,5} |
| 6 | (1,-1) | {1,2,3} |
| 7 | (1,0) | {3,4,5} |
| 8 | (2,-3) | {0,2,3} |
| 9 | (2,-2) | {0,1,5} |
| 10 | (2,-1) | {2,4,5} |
| 11 | (3,-3) | {1,2,3} |
| 12 | (3,-2) | {2,3,4} |

### AD witness

| Site | Axial coordinate | Patch directions |
|---:|:---:|:---:|
| 1 | (-3,2) | {0,1,5} |
| 2 | (-3,3) | {0,4,5} |
| 3 | (-2,0) | {0,1,5} |
| 4 | (-2,1) | {0,2,4} |
| 5 | (-2,2) | {1,2,3} |
| 6 | (-2,3) | {3,4,5} |
| 7 | (-1,-1) | {0,1,2} |
| 8 | (-1,0) | {3,4,5} |
| 9 | (-1,1) | {1,3,5} |
| 10 | (-1,2) | {0,2,4} |
| 11 | (0,-1) | {1,2,3} |
| 12 | (0,0) | {0,2,4} |
| 13 | (0,1) | {0,1,5} |
| 14 | (0,2) | {3,4,5} |
| 15 | (1,0) | {1,2,3} |
| 16 | (1,1) | {2,3,4} |
