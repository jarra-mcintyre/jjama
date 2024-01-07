package me.jmcintyre;

import com.carrotsearch.hppc.IntArrayList;

import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.CharBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.text.Normalizer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

final class Tokenizer {
    public static final int UNKNOWN_ID = 0;
    public static final int BEGINNING_OF_SEQUENCE_ID = 1;
    public static final int END_OF_SEQUENCE_ID = 2;

    private final String[] vocab;
    private final float[] vocabScores;
    private final Map<CharBuffer, Integer> tokenIds;
    private final int maxTokenLength;

    public static Tokenizer loadBin(Path path, int vocabSize) throws IOException {
        var vocab = new String[vocabSize];
        var vocabScores = new float[vocabSize];
        try (var channel = FileChannel.open(path, StandardOpenOption.READ)) {
            var map = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size()).order(ByteOrder.nativeOrder());
            map.position(4); // skip max token length
            for (int i = 0; i < vocabSize; i++) {
                vocabScores[i] = map.getFloat();
                int length = map.getInt();
                byte[] utf8 = new byte[length];
                map.get(utf8);
                vocab[i] = new String(utf8, StandardCharsets.UTF_8);
            }
        }
        return new Tokenizer(vocab, vocabScores);
    }

    public Tokenizer(String[] vocab, float[] vocabScores) {
        this.vocab = vocab;
        this.vocabScores = vocabScores;
        this.tokenIds = IntStream.range(0, vocab.length)
                .mapToObj(i -> Map.entry(CharBuffer.wrap(vocab[i]), i))
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
        this.maxTokenLength = Arrays.stream(vocab).mapToInt(String::length).max().orElse(0);
    }

    /**
     * <p>This makes a basic attempt to emulate the behaviour of the SentencePiece encode function.</p>
     *
     * <p>In SentencePiece whitespace is normalised to use '▁' (unicode \\u2581) is used for whitespace. Here it is not.</p>
     *
     * @param target target string to encoded
     * @param bos    add a beginning of sequence mark at the start of the encoded output?
     * @param eos    add an end of sequence mark at the end of the encoded output?
     * @return list of token ids
     */
    IntArrayList encode(String target, boolean leadingSpace, boolean bos, boolean eos) {
        //var normalized = Normalizer.normalize(target, Normalizer.Form.NFKC);
        var normalized = target;  // normalisation is not applied
        var tokens = new IntArrayList();

        if (bos) {
            tokens.add(BEGINNING_OF_SEQUENCE_ID);
        }

        if (leadingSpace && !normalized.isEmpty()) {
            tokens.add(tokenIds.get(CharBuffer.wrap(" ")));
        }

        var buffer = CharBuffer.allocate(maxTokenLength);
        normalized.codePoints().forEach(codePoint -> {
            buffer.clear();
            if (Character.isBmpCodePoint(codePoint)) {
                buffer.put((char) codePoint);
            } else {
                buffer.put(Character.highSurrogate(codePoint));
                buffer.put(Character.lowSurrogate(codePoint));
            }
            Integer id = tokenIds.get(buffer.flip());
            if (id != null) {
                tokens.add(id);
            } else {
                // Split into UTF-8 code points and add each as a separate byte token (that is tokens 3 to 258)
                for (byte b : Character.toString(codePoint).getBytes(StandardCharsets.UTF_8)) {
                    tokens.add((int) b + 3);
                }
            }
        });

        // greedily merge neighbouring tokens
        while (true) {
            float bestScore = Float.NEGATIVE_INFINITY;
            int bestIndex = -1, bestId = -1;
            for (int i = 0; i < tokens.size() - 1; i++) {
                buffer.clear();
                buffer.put(vocab[tokens.get(i)]);
                buffer.put(vocab[tokens.get(i + 1)]);
                Integer id = tokenIds.get(buffer.flip());
                if (id != null && vocabScores[id] > bestScore) {
                    bestScore = vocabScores[id];
                    bestId = id;
                    bestIndex = i;
                }
            }

            if (bestIndex == -1) {
                break;
            }

            tokens.set(bestIndex, bestId);
            tokens.remove(bestIndex + 1);
        }

        if (eos) {
            tokens.add(END_OF_SEQUENCE_ID);
        }

        return tokens;
    }

    public boolean isEndOfSequence(int id) {
        return id == END_OF_SEQUENCE_ID;
    }

    public String decodeToken(int id) {
        // First 3 vocab items are unknown/beginning of sequence/end of sequence
        if (id > 255 + 3) {
            return vocab[id];
        } else if (id > 3) {
            return String.valueOf((char)(id - 3));
        } else {
            return "";
        }
    }

    public String decodeTokenString(IntArrayList ids) {
        return Arrays.stream(ids.buffer, 0, ids.size()).mapToObj(this::decodeToken).collect(Collectors.joining());
    }

    @Override
    public String toString() {
        return STR."Tokenizer{vocab=[\{vocab.length}]\{'}'}";
    }
}
