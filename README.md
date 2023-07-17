# SymMatNP

---

## Symmetric Matrices in Numpy

This is a memory-efficient implementation to work with symmetric matrices in numpy-style. The backbone of this is a 
one dimensional numpy array storing the upper triangle of a symmetric matrix. The diagonal values are stored as a 
single value as we require them to be the same, i.e. the main diagonal of these matrices has to have the same value in 
every spot.
