# lvm/egifa/pack

Quick test to see if PACK(d=1, different J) behaves differently

- Higher J means higher resolution, more sharp features at same M (M does the same)
- Therefore we can trade M (costly) for J (for free once amortized) for free
- Weights of PACK kernel (ie normalized sigma params) decay naturally
- Both for J=1 and J=8 we see that clusters _are_ learned once Q >= 3
- J improves fit and decreases OQ sensitivity to fit

K=4 clusters seems enough at any rate

qGPLVM works! PACK:1 works!
