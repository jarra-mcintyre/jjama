package me.jmcintyre;

import com.carrotsearch.hppc.sorting.IndirectSort;

import java.util.Random;
import java.util.stream.IntStream;

class Sampler {
    Random random;
    float topPCutoff;

    int[] indices;

    public Sampler(int vocabSize, float topPCutoff, int seed) {
        this.random = new Random(seed);
        this.topPCutoff = topPCutoff;
        this.indices = IntStream.range(0, vocabSize).toArray();
    }

    int sampleTopP(float[] probabilities, float sampleProbability) {
        IndirectSort.mergesort(indices, (i1, i2) -> -Float.compare(probabilities[i1], probabilities[i2]));
        //return indices[0];
        // Find the top N candidates. They will have a cumulative 'probability' gte topP
        float topNProbability = 0.f;
        int cutoffIndex = 0;
        for (; cutoffIndex < indices.length; cutoffIndex++) {
            topNProbability += probabilities[indices[cutoffIndex]];
            if (topNProbability > topPCutoff) {
                cutoffIndex += 1;
                break;
            }
        }
        float sampleCutoff = topNProbability * sampleProbability, sampleCdf = 0;
        for (int i = 0; i < cutoffIndex; i++) {
            sampleCdf += probabilities[indices[i]];
            if (sampleCdf >= sampleCutoff) {
                return indices[i];
            }
        }
        return indices[cutoffIndex - 1];
    }

    int sample(float[] probabilities) {
        return sampleTopP(probabilities, random.nextFloat());
    }
}
