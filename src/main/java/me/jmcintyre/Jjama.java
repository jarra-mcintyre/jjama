package me.jmcintyre;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;
import java.util.Random;

public class Jjama {
    static void generate(Model model, Tokenizer tokenizer, Sampler sampler, String prompt) {
        var context = new ModelInferenceContext(model);
        var tokens = tokenizer.encode(Objects.requireNonNullElse(prompt, ""), true, true, false);
        long elapsed = 0, tokensProcessed = 0;
        for (int pos = 0, previousToken = 0; pos < model.config().seqLen(); pos++) {
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

            if (tokenizer.isEndOfSequence(output) && pos > 0) {
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
        var model = Model.readBinFile(modelPath);
        var tokenizer = Tokenizer.loadBin(tokenizerPath, model.config().vocabSize());
        var sampler = new Sampler(model.config().vocabSize(), 0.9f, (new Random()).nextInt());
        generate(model, tokenizer, sampler, null);
    }
}
