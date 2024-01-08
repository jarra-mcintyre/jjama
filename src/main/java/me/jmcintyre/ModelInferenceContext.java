package me.jmcintyre;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.util.Arrays;
import java.util.stream.IntStream;

class ModelInferenceContext {
    private final ModelConfig config;
    private final TransformerWeights weights;

    // Inference state
    private final float[] x, xb; // (dim,)
    private final float[] hb, hb2; // (hidden_dim,)
    private final float[] q; // (dim,)
    private final float[] result; // (vocab_size,)
    private final float[][][] keyCache, valueCache; // (layer, seq_len, kv_dim)

    public ModelInferenceContext(Model model) {
        this.config = model.config();
        this.weights = model.weights();
        this.x = new float[config.dim()];
        this.xb = new float[config.dim()];
        this.hb = new float[config.hiddenDim()];
        this.hb2 = new float[config.hiddenDim()];
        this.q = new float[config.dim()];
        this.result = new float[config.vocabSize()];
        this.keyCache = IntStream.range(0, config.nLayers()).mapToObj(_ -> new float[config.seqLen()][]).toArray(float[][][]::new);
        this.valueCache = IntStream.range(0, config.nLayers()).mapToObj(_ -> new float[config.seqLen()][]).toArray(float[][][]::new);
    }

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    static float dot(float[] x, float[] y, int xOffset, int yOffset, int size) {
        final int upperBound = SPECIES.loopBound(size);
        int j = 0;
        var acc = FloatVector.zero(SPECIES);
        for (; j < upperBound; j += SPECIES.length()) {
            var a = FloatVector.fromArray(SPECIES, x, xOffset + j);
            var b = FloatVector.fromArray(SPECIES, y, yOffset + j);
            acc = a.fma(b, acc);
        }
        float dot = acc.reduceLanes(VectorOperators.ADD);
        for (; j < size; j++) {
            dot += x[j] * y[j];
        }
        return dot;
    }


    static void mmul(float[] output, float[] v, float[] m, int sharedDim, int resultDim) {
        for (int i = 0; i < resultDim; i++) {
            output[i] = dot(m, v, i * sharedDim, 0, sharedDim);
        }
    }

    static void addMMulInto(float[] target, float[] v, float[] m, int sharedDim, int resultDim) {
        for (int i = 0; i < resultDim; i++) {
            target[i] += dot(m, v, i * sharedDim, 0, sharedDim);
        }
    }

    /**
     * RMS Normal storing the result into output
     */
    static void rmsNormal(float[] output, float[] input, float[] weights) {
        double ss = 0;
        for (float v : input) {
            ss += v * v;
        }
        float scale = (float) (1 / Math.sqrt(ss / input.length + 1e-5));
        for (int i = 0; i < input.length; i++) {
            output[i] = weights[i] * (scale * input[i]);
        }
    }

    /**
     * softmax operator applied in-place to target
     */
    static void inplaceSoftmax(float[] target) {
        float exponentBase = Float.NEGATIVE_INFINITY;
        for (float v : target) {
            exponentBase = Math.max(exponentBase, v);
        }
        double sum = 0;
        for (int i = 0; i < target.length; i++) {
            double v = Math.exp(target[i] - exponentBase);
            sum += v;
            target[i] = (float) v;
        }
        for (int i = 0; i < target.length; i++) {
            target[i] /= (float) sum;
        }
    }

    /**
     * RoPE positional embedding applied in-place to target
     */
    static void inplaceRope(float[] target, int pos, int headSize, double baseWaveLength) {
        for (int i = 0; i < target.length; i += 2) {
            int headDim = i % headSize;
            double freq = 1.0 / (float) Math.pow(baseWaveLength, headDim / (double) headSize);
            double value = pos * freq;
            float fcr = (float) Math.cos(value), fci = (float) Math.sin(value);
            float v0 = target[i], v1 = target[i + 1];
            target[i] = v0 * fcr - v1 * fci;
            target[i + 1] = v0 * fci + v1 * fcr;
        }
    }

    public float[] forward(int token, int pos, boolean outputNextTokenPrediction) {
        final float sqrtHeadSize = (float) Math.sqrt(config.headSize());
        System.arraycopy(weights.tokenEmbeddingTable(), token * config.dim(), x, 0, config.dim());
        for (int l = 0; l < config.nLayers(); l++) {
            rmsNormal(xb, x, weights.rmsAttWeight()[l]);

            // populate key cache for position
            keyCache[l][pos] = new float[config.kvDim()];
            mmul(keyCache[l][pos], xb, weights.wk()[l], config.dim(), config.kvDim());
            inplaceRope(keyCache[l][pos], pos, config.headSize(), 10000.0);
            // populate value cache for position
            valueCache[l][pos] = new float[config.kvDim()];
            mmul(valueCache[l][pos], xb, weights.wv()[l], config.dim(), config.kvDim());

            // When processing the prompt it's only necessary to prime the KV cache.
            if (!outputNextTokenPrediction && l == config.nLayers() - 1) {
                return null;
            }

            // derive the query
            mmul(q, xb, weights.wq()[l], config.dim(), config.dim());
            inplaceRope(q, pos, config.headSize(), 10000.0);

            Arrays.fill(xb, 0);
            for (int h = 0; h < config.nHeads(); h++) {
                int headOffset = h * config.headSize();
                int kvHeadOffset = h / (config.nHeads() / config.nKVHeads()) * config.headSize();

                // https://arxiv.org/pdf/2112.05682.pdf
                float sumScore = 0, expBase = Float.NEGATIVE_INFINITY;
                for (int t = 0; t <= pos; t++) {
                    float[] k = keyCache[l][t], v = valueCache[l][t];
                    float score = dot(q, k, headOffset, kvHeadOffset, config.headSize()) / sqrtHeadSize;

                    float stepExpBase = Math.max(score, expBase);
                    float wu = (float) Math.exp(expBase - stepExpBase), wv = (float) Math.exp(score - stepExpBase);

                    expBase = stepExpBase;
                    sumScore = wu * sumScore + wv;
                    for (int i = 0; i < config.headSize(); i++) {
                        xb[headOffset + i] = wu * xb[headOffset + i] + wv * v[headOffset + i];
                    }
                }

                for (int i = 0; i < config.headSize(); i++) {
                    xb[headOffset + i] /= sumScore;
                }
            }

            // x = x + wo[l] @ xb
            addMMulInto(x, xb, weights.wo()[l], config.dim(), config.dim());
            rmsNormal(xb, x, weights.rmsFFNWeight()[l]);

            // w2 * (w3 * x) * silu(w1*x)
            mmul(hb, xb, weights.w1()[l], config.dim(), config.hiddenDim());
            mmul(hb2, xb, weights.w3()[l], config.dim(), config.hiddenDim());

            for (int i = 0; i < hb.length; i++) {
                float val = hb[i];
                val *= (1.0f / (1.0f + (float) Math.exp(-val))); // silu(x) = x*sigma(x)
                hb[i] = val * hb2[i];
            }

            // x = x + w2@hb
            addMMulInto(x, hb, weights.w2()[l], config.hiddenDim(), config.dim());
        }

        rmsNormal(xb, x, weights.rmsFinalWeight());
        mmul(result, xb, weights.wcls(), config.dim(), config.vocabSize());
        inplaceSoftmax(result);
        return result;
    }
}
