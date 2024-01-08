package me.jmcintyre;

import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.function.Supplier;

/**
 * @param tokenEmbeddingTable (vocab_size, dim)
 * @param rmsAttWeight        (layer, dim)
 * @param rmsFFNWeight        (layer, dim)
 * @param wq                  (layer, dim, n_heads * head_size)
 * @param wk                  (layer, dim, n_kv_heads * head_size)
 * @param wv                  (layer, dim, n_kv_heads * head_size)
 * @param wo                  (layer, n_heads * head_size, dim)
 * @param w1                  (layer, hidden_dim, dim)
 * @param w2                  (layer, dim, hidden_dim)
 * @param w3                  (layer, hidden_dim, dim)
 * @param rmsFinalWeight      (dim,)
 */
record TransformerWeights(
        float[] tokenEmbeddingTable,
        float[][] rmsAttWeight,
        float[][] wq,
        float[][] wk,
        float[][] wv,
        float[][] wo,
        float[][] rmsFFNWeight,
        float[][] w1,
        float[][] w2,
        float[][] w3,
        float[] rmsFinalWeight,
        float[] wcls
) { }
