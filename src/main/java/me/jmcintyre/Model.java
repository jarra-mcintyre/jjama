package me.jmcintyre;

import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.function.Supplier;

public record Model(
        ModelConfig config,
        TransformerWeights weights
) {

    private static float[][] readShardedTensor(FloatBuffer map, int d1, int d2, int d3) {
        return readShardedTensor(map, d1, d2 * d3);
    }

    private static float[][] readShardedTensor(FloatBuffer map, int shardDim, int otherDim) {
        float[][] result = new float[shardDim][];
        ;
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

    public static Model readBinFile(Path path) throws IOException {
        try (var channel = FileChannel.open(path, StandardOpenOption.READ)) {
            // read configuration header
            var map = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size()).order(ByteOrder.nativeOrder());
            var intMap = map.asIntBuffer();
            var config = new ModelConfig(intMap.get(), intMap.get(), intMap.get(), intMap.get(), intMap.get(), intMap.get(), intMap.get());
            // read weights
            map.position(intMap.position() * 4);
            var floatMap = map.asFloatBuffer();
            var tokenEmbeddingTable = readTensor(floatMap, config.vocabSize(), config.dim());
            var weights = new TransformerWeights(
                    tokenEmbeddingTable,
                    readShardedTensor(floatMap, config.nLayers(), config.dim()),
                    readShardedTensor(floatMap, config.nLayers(), config.dim(), config.nHeads() * config.headSize()),
                    readShardedTensor(floatMap, config.nLayers(), config.dim(), config.nKVHeads() * config.headSize()),
                    readShardedTensor(floatMap, config.nLayers(), config.dim(), config.nKVHeads() * config.headSize()),
                    readShardedTensor(floatMap, config.nLayers(), config.nHeads() * config.headSize(), config.dim()),
                    readShardedTensor(floatMap, config.nLayers(), config.dim()),
                    readShardedTensor(floatMap, config.nLayers(), config.dim(), config.hiddenDim()),
                    readShardedTensor(floatMap, config.nLayers(), config.hiddenDim(), config.dim()),
                    readShardedTensor(floatMap, config.nLayers(), config.dim(), config.hiddenDim()),
                    readTensor(floatMap, config.dim()),
                    config.sharedWeights() ? tokenEmbeddingTable : ((Supplier<float[]>) () -> {
                        floatMap.position(floatMap.position() + config.seqLen() * config.headSize()); // skip unused
                        return readTensor(floatMap, config.vocabSize(), config.dim());
                    }).get()
            );
            return new Model(config, weights);
        }
    }
}
