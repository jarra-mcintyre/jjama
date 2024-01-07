package me.jmcintyre;

import com.carrotsearch.hppc.IntArrayList;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class TokenizerTest {
    @Test
    void testEncode() throws IOException {
        var tokenizer = Tokenizer.loadBin(Path.of("tokenizer.bin"), 32000);
        var testStrings = List.of(
                "This is a   \ttest string",
                "　　これはテスト用の文字列です。"
        );
        var expectedEncodings = List.of(
                IntArrayList.from(910, 338, 263, 1678, 12, 1688, 1347),
                IntArrayList.from(29871, 30358, 30358, 30589, 30553, 30449, 30572, 30255, 30279, 30406, 30199, 30333, 30578, 31025, 30499, 30427, 30267)
        );

        for (int i = 0; i < testStrings.size(); i++) {
            assertEquals(expectedEncodings.get(i), tokenizer.encode(testStrings.get(i), true, false, false));
        }

        for (var testString : testStrings) {
            assertEquals(testString, tokenizer.decodeTokenString(tokenizer.encode(testString, false, false, false)));
        }
    }
}
