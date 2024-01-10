package me.jmcintyre;

import java.util.Random;

class Sampler {
    final Random random;
    final float topPCutoff;
    final int vocabSize;

    final int[] indices;
    final int[] sortedIndices;

    public Sampler(int vocabSize, float topPCutoff, int seed) {
        this.random = new Random(seed);
        this.topPCutoff = topPCutoff;
        this.vocabSize = vocabSize;
        this.indices = new int[vocabSize];
        this.sortedIndices = new int[vocabSize];
    }

    private static void argSort(float[] values, int[] source, int[] destination, int offset, int length) {
        assert source != destination;
        System.arraycopy(source, offset, destination, offset, length);
        mergeArgSort(values, source, destination, offset, offset + length);
    }

    private static void mergeArgSort(float[] values, int[] source, int[] destination, int fromIndex, int toIndex) {
        if (toIndex - fromIndex <= 32) {
            insertionArgSort(values, destination, fromIndex, toIndex);
        } else {
            int midIndex = fromIndex + toIndex >>> 1;
            mergeArgSort(values, destination, source, fromIndex, midIndex);
            mergeArgSort(values, destination, source, midIndex, toIndex);

            if (values[source[midIndex - 1]] > values[source[midIndex]]) {
                // subsets are already sorted
                System.arraycopy(source, fromIndex, destination, fromIndex, toIndex - fromIndex);
            } else {
                // merge sorted subsets into destination
                int lowerIndex = fromIndex, upperIndex = midIndex;
                for (int i = fromIndex; i < toIndex; i++) {
                    if (upperIndex != toIndex && (lowerIndex >= midIndex || values[source[lowerIndex]] < values[source[upperIndex]])) {
                        destination[i] = source[upperIndex++];
                    } else {
                        destination[i] = source[lowerIndex++];
                    }
                }
            }
        }
    }

    private static void insertionArgSort(float[] values, int[] indices, int fromIndex, int toIndex) {
        for (int i = fromIndex + 1; i < toIndex; i++) {
            int v =  indices[i], j = i;
            while (j > fromIndex) {
                int t = indices[j - 1];
                if (values[v] > values[t]) {
                    indices[j--] = t;
                } else {
                    break;
                }
            }
            indices[j] = v;
        }
    }

    int sampleTopP(float[] probabilities, float sampleProbability) {
        float minimumProbability = (1 - topPCutoff) / (vocabSize - 1);
        int candidates = 0;
        for (int i = 0; i < probabilities.length; i++) {
            if (probabilities[i] >= minimumProbability) {
                indices[candidates++] = i;
            }
        }

        // Quick sort candidates
        argSort(probabilities, indices, sortedIndices, 0, candidates);

        // Find the top N candidates. They will have a cumulative 'probability' gte topP
        float topNProbability = 0.f;
        int cutoffIndex = 0;
        for (; cutoffIndex < candidates; cutoffIndex++) {
            topNProbability += probabilities[sortedIndices[cutoffIndex]];
            if (topNProbability > topPCutoff) {
                cutoffIndex += 1;
                break;
            }
        }

        float sampleCutoff = topNProbability * sampleProbability, sampleCdf = 0;
        for (int i = 0; i < cutoffIndex; i++) {
            sampleCdf += probabilities[sortedIndices[i]];
            if (sampleCdf >= sampleCutoff) {
                return sortedIndices[i];
            }
        }

        return sortedIndices[cutoffIndex - 1];
    }

    int sample(float[] probabilities) {
        return sampleTopP(probabilities, random.nextFloat());
    }
}
