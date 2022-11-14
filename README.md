# CS177H_MSAs_Project
> Classify Multiple Sequence Alignments (MSAs) for AlphaFold2

Recently, AlphaFold2 from Google DeepMind has made groundbreaking performance in the protein
folding competition of CASP 14. The quality of Multiple Sequence Alignments (MSAs), a key input data
for AlphaFold2, has a significant impact on AlphaFold2's performance. If better MSAs are built,
AlphaFold2 can predict structures more accurately. However, there exists no reliable method as yet to judge
which MSA is better. In this project, you are expected to train a neural network to predict if a given MSA
is of high quality for AlphaFold2. It can be formulated as a binary classification problem. You are
recommended to use MSA Transformer [2] as an encoder to choose the better MSA from a pair of MSAs
accurately. Training data and test data set will be provided

## References:
- [1] Jumper, J., Evans, R., Pritzel, A. et al. Highly accurate protein structure prediction with AlphaFold. Nature 596, 583–589, 2021. https://doi.org/10.1038/s41586-021-03819-2
- [2] Rao R, Liu J, Verkuil R, Meier J, Canny JF, Abbeel P, et al. MSA Transformer. 2021. https://doi.org/10.1101/2021.02.12.430858
- [3] https://predictioncenter.org/casp14/doc/presentations/
