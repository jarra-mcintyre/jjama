package me.jmcintyre;

import com.carrotsearch.hppc.sorting.IndirectSort;

import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Objects;
import java.util.Random;
import java.util.stream.IntStream;

public class Jjama {
    record Config(
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

    static class TransformerWeights {
        float[] tokenEmbeddingTable; // (vocab_size, dim)
        float[][] rmsAttWeight; // (layer, dim)
        float[][] rmsFFNWeight; // (layer, dim)
        float[][] wq; // (layer, dim, n_heads * head_size)
        float[][] wk; // (layer, dim, n_kv_heads * head_size)
        float[][] wv; // (layer, dim, n_kv_heads * head_size)
        float[][] wo; // (layer, n_heads * head_size, dim)
        float[][] w1; // (layer, hidden_dim, dim)
        float[][] w2; // (layer, dim, hidden_dim)
        float[][] w3; // (layer, hidden_dim, dim)
        float[] rmsFinalWeight; // (dim,)
        float[] wcls;
    }

    static class Sampler {
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
            for (;  cutoffIndex < indices.length; cutoffIndex++) {
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

    public record Model(Config config, TransformerWeights weights) { }

    static class ModelInferenceContext {
        Config config;
        TransformerWeights weights;

        // Inference state
        float[] x, xb; // (dim,)
        float[] hb, hb2; // (hidden_dim,)
        float[] q; // (dim,)
        float[] result; // (vocab_size,)
        float[][][] keyCache, valueCache; // (layer, seq_len, kv_dim)
        float topP;

        public ModelInferenceContext(Model model) {
            this.config = model.config();
            this.weights = model.weights();
            this.x = new float[config.dim];
            this.xb = new float[config.dim];
            this.hb = new float[config.hiddenDim];
            this.hb2 = new float[config.hiddenDim];
            this.q = new float[config.dim];
            this.result = new float[config.vocabSize()];
            this.keyCache = IntStream.range(0,config.nLayers).mapToObj(_ -> new float[config.seqLen][]).toArray(float[][][]::new);
            this.valueCache = IntStream.range(0,config.nLayers).mapToObj(_ -> new float[config.seqLen][]).toArray(float[][][]::new);
        }

        static void mmul(float[] output, float[] v, float[] m, int sharedDim, int resultDim) {
            for (int i = 0; i < resultDim; i++) {
                float dot = 0;
                for (int j = 0; j < sharedDim; j++) {
                    dot += m[i * sharedDim + j] * v[j];
                }
                output[i] = dot;
            }
        }

        static void addMMulInto(float[] target, float[] v, float[] m, int sharedDim, int resultDim) {
            for (int i = 0; i < resultDim; i++) {
                float dot = 0;
                for (int j = 0; j < sharedDim; j++) {
                    dot += m[i * sharedDim + j] * v[j];
                }
                target[i] += dot;
            }
        }

        /**
         * RMS Normal storing the result into output
         */
        static void rmsNormal(float[] output, float[] input, float[] weights) {
            double ss = 0;
            for (float v : input) {
                ss += v*v;
            }
            float scale = (float)(1 / Math.sqrt(ss / input.length + 1e-5));
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
                double freq = 1.0 / (float)Math.pow(baseWaveLength, headDim / (double)headSize);
                double value = pos * freq;
                float fcr = (float) Math.cos(value), fci = (float) Math.sin(value);
                float v0 = target[i], v1 = target[i + 1];
                target[i] = v0 * fcr - v1 * fci;
                target[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        public float[] forward(int token, int pos, boolean outputNextTokenPrediction) {
            final float sqrtHeadSize = (float) Math.sqrt(config.headSize());
            System.arraycopy(weights.tokenEmbeddingTable, token * config.dim, x, 0, config.dim);
            for (int l = 0; l < config.nLayers; l++) {
                rmsNormal(xb, x, weights.rmsAttWeight[l]);

                // populate key cache for position
                keyCache[l][pos] = new float[config.kvDim()];
                mmul(keyCache[l][pos], xb, weights.wk[l], config.dim, config.kvDim());
                inplaceRope(keyCache[l][pos], pos, config.headSize(), 10000.0);
                // populate value cache for position
                valueCache[l][pos] = new float[config.kvDim()];
                mmul(valueCache[l][pos], xb, weights.wv[l], config.dim, config.kvDim());

                // When processing the prompt it's only necessary to prime the KV cache.
                if (!outputNextTokenPrediction && l == config.nLayers - 1) {
                    return null;
                }

                // derive the query
                mmul(q, xb, weights.wq[l], config.dim, config.dim);
                inplaceRope(q, pos, config.headSize(), 10000.0);

                Arrays.fill(xb, 0);
                for (int h = 0; h < config.nHeads; h++) {
                    int headOffset = h * config.headSize();
                    int kvHeadOffset = h / (config.nHeads / config.nKVHeads) * config.headSize();

                    // https://arxiv.org/pdf/2112.05682.pdf
                    float sumScore = 0, expBase = Float.NEGATIVE_INFINITY;
                    for (int t = 0; t <= pos; t++) {
                        float[] k = keyCache[l][t], v = valueCache[l][t];
                        float score = 0.0f;
                        for (int i = 0; i < config.headSize(); i++) {
                            score += q[headOffset + i] * k[kvHeadOffset + i];
                        }
                        score /= sqrtHeadSize;

                        float stepExpBase = Math.max(score, expBase);
                        float wu = (float) Math.exp(expBase - stepExpBase), wv = (float)Math.exp(score - stepExpBase);

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
                addMMulInto(x, xb, weights.wo[l], config.dim, config.dim);
                rmsNormal(xb, x, weights.rmsFFNWeight[l]);

                // w2 * (w3 * x) * silu(w1*x)
                mmul(hb, xb, weights.w1[l], config.dim, config.hiddenDim);
                mmul(hb2, xb, weights.w3[l], config.dim, config.hiddenDim);

                for (int i = 0; i < hb.length; i++) {
                    float val = hb[i];
                    val *= (1.0f / (1.0f + (float)Math.exp(-val))); // silu(x) = x*sigma(x)
                    hb[i] = val * hb2[i];
                }

                // x = x + w2@hb
                addMMulInto(x, hb, weights.w2[l], config.hiddenDim, config.dim);
            }

            rmsNormal(xb, x, weights.rmsFinalWeight);
            mmul(result, xb, weights.wcls, config.dim, config.vocabSize());
            inplaceSoftmax(result);
            return result;
        }
    }

    private static float[][] readShardedTensor(FloatBuffer map, int d1, int d2, int d3) {
        return readShardedTensor(map, d1, d2 * d3);
    }

    private static float[][] readShardedTensor(FloatBuffer map, int shardDim, int otherDim) {
        float[][] result = new float[shardDim][];;
        for (int i = 0; i < shardDim; i++) {
            result[i] = new float[otherDim];
            map.get(result[i]);
        }
        return result;
    }

    private static float[] readTensor(FloatBuffer map, int dim1, int dim2) {
        return readTensor(map, dim1 * dim2);
    }

    private static float[] readTensor(FloatBuffer map, int length) {
        float[] result = new float[length];
        map.get(result);
        return result;
    }

    static Model readModel(Path path) throws IOException {
        try (var channel = FileChannel.open(path, StandardOpenOption.READ)) {
            // read configuration header
            var map = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size()).order(ByteOrder.nativeOrder());
            var intMap = map.asIntBuffer();
            var config = new Config(intMap.get(), intMap.get(), intMap.get(), intMap.get(), intMap.get(), intMap.get(), intMap.get());
            // read weights
            map.position(intMap.position() * 4);
            var floatMap = map.asFloatBuffer();
            var weights = new TransformerWeights();
            weights.tokenEmbeddingTable = readTensor(floatMap, config.vocabSize(), config.dim);
            weights.rmsAttWeight = readShardedTensor(floatMap, config.nLayers, config.dim);
            weights.wq = readShardedTensor(floatMap, config.nLayers, config.dim, config.nHeads * config.headSize());
            weights.wk = readShardedTensor(floatMap, config.nLayers, config.dim, config.nKVHeads * config.headSize());
            weights.wv = readShardedTensor(floatMap, config.nLayers, config.dim, config.nKVHeads * config.headSize());
            weights.wo = readShardedTensor(floatMap, config.nLayers, config.nHeads * config.headSize(), config.dim);
            weights.rmsFFNWeight = readShardedTensor(floatMap, config.nLayers, config.dim);
            weights.w1 = readShardedTensor(floatMap, config.nLayers, config.dim, config.hiddenDim);
            weights.w2 = readShardedTensor(floatMap, config.nLayers, config.hiddenDim, config.dim);
            weights.w3 = readShardedTensor(floatMap, config.nLayers, config.dim, config.hiddenDim);
            weights.rmsFinalWeight = readTensor(floatMap, config.dim);
            floatMap.position(floatMap.position() + config.seqLen * config.headSize()); // skip unused
            weights.wcls = config.sharedWeights() ? weights.tokenEmbeddingTable : readTensor(floatMap, config.vocabSize(), config.dim);
            return new Model(config, weights);
        }
    }

    static void generate(Model model, Tokenizer tokenizer, Sampler sampler, String prompt) {
        var context = new ModelInferenceContext(model);
        var tokens = tokenizer.encode(Objects.requireNonNullElse(prompt, ""), true, true, false);
        long elapsed = 0, tokensProcessed = 0;
        for (int pos = 0, previousToken = 0; pos < context.config.seqLen; pos++) {
            long start = System.nanoTime();

            int output;
            if (pos < tokens.size()) {
                context.forward(tokens.get(pos), pos, false);
                output = tokens.get(pos);
            } else {
                float[] tokenProbabilities = context.forward(previousToken, pos, true);
                output = sampler.sample(tokenProbabilities);
            }

            if (pos >= 5) { // allow a few iterations for the jit to warm-up before measuring output times
                elapsed += System.nanoTime() - start;
                tokensProcessed += 1;
            }

            if (tokenizer.isEndOfSequence(output)) {
                break;
            } else {
                System.out.print(tokenizer.decodeToken(output));
                previousToken = output;
            }
        }

        double tokensPerSecond = tokensProcessed / (elapsed / 1e9);
        System.out.printf("%n%n%.3f tokens per second%n", tokensPerSecond);
    }

    public static void main(String[] args) throws IOException {
        var modelPath = Path.of("stories15M.bin");
        var tokenizerPath = Path.of("tokenizer.bin");
        var model = readModel(modelPath);
        var tokenizer = Tokenizer.loadBin(tokenizerPath, model.config.vocabSize());
        var sampler = new Sampler(model.config.vocabSize(), 0.9f, 1337);
        generate(model, tokenizer, sampler, null);
    }
}
