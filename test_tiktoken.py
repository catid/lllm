import unittest
import tiktoken

class TestTikTokenizer(unittest.TestCase):
    def setUp(self):
        # Initialize the tokenizer with the latest model
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")

    def generate_test_cases(self):
        # Programmatically generate a list of test cases
        cases = []
        
        # Manually curated examples
        curated_examples = [
            "hello world",
            "the quick brown fox jumps over the lazy dog",
            "1234567890",
            "!@#$%^&*()_+-=[]{};':,./<>?",
            "long text " * 10,
            "new\nlines\nincluded",
            "\t tabs \t and \n new lines",
            "MixedCASE and special $ymbols together",
            "Emoji 😊 support?",
            "Numbers 123 and letters ABC together",
            "Spaces and tabs\t\ttogether",
            "Very long string " + "a" * 100 + " with padding",
            "Ends with newline\n",
            "\nStarts with newline",
            "Contains unicode: üöäÜÖÄß",
            "Random text with newline\nin the middle"
        ]
        cases.extend(curated_examples)

        # Generate cases with repeating patterns
        for i in range(10):
            repeat_str = f"repeat{i} " * (i + 1)
            cases.append(repeat_str.strip())

        # Generate cases with increasing length
        for i in range(10, 110, 10):
            increasing_str = ''.join(chr((x % 26) + 97) for x in range(i))
            cases.append(increasing_str)

        # Random mix of characters
        import random
        import string
        for _ in range(30):  # Generate 30 random cases
            random_length = random.randint(1, 100)
            random_str = ''.join(random.choices(string.ascii_letters + string.digits + string.punctuation + " ", k=random_length))
            cases.append(random_str)

        return cases

    def test_multiple_encode_decode_trials(self):
        test_cases = self.generate_test_cases()
        for original_str in test_cases:
            with self.subTest(original_str=original_str):
                encoded_tokens = self.tokenizer.encode(original_str)
                decoded_str = self.tokenizer.decode(encoded_tokens)
                self.assertEqual(original_str, decoded_str, f"Decoded string does not match the original for input '{original_str}'")

    def test_multiple_encodeBytes_decodeBytes_trials(self):
        test_cases = self.generate_test_cases()
        for original_str in test_cases:
            original_bytes = original_str.encode('utf-8')
            with self.subTest(original_bytes=original_bytes):
                encoded_tokens = self.tokenizer.encode(original_str)
                decoded_str = self.tokenizer.decode(encoded_tokens)
                decoded_bytes = decoded_str.encode('utf-8')
                self.assertEqual(original_bytes, decoded_bytes, f"Decoded bytes do not match the original for input '{original_bytes}'")

if __name__ == '__main__':
    unittest.main()
