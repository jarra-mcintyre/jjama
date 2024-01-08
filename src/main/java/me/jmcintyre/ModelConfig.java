package me.jmcintyre;

record ModelConfig(
        int dim,
        int hiddenDim,
        int nLayers,
        int nHeads,
        int nKVHeads,
        int vocabSizeRaw,
        int seqLen
) {
    int headSize() {
        return dim / nHeads;
    }

    boolean sharedWeights() {
        return vocabSizeRaw > 0; // sign on vocab size indicates if weights are shared
    }

    int vocabSize() {
        return Math.abs(vocabSizeRaw);
    }

    int kvDim() {
        return (dim * nKVHeads) / nHeads;
    }
}
