import string
from collections import defaultdict, Counter


def preprocess_text(text):
    """
    Convert text to lowercase, remove punctuation, and split into tokens.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


def build_trigram_counts(tokens):
    """
    Build trigram counts:
    (word1, word2) -> Counter({next_word: count})
    """
    trigram_counts = defaultdict(Counter)

    for i in range(len(tokens) - 2):
        context = (tokens[i], tokens[i + 1])
        next_word = tokens[i + 2]
        trigram_counts[context][next_word] += 1

    return trigram_counts


def build_trigram_probabilities(trigram_counts):
    """
    Convert trigram counts to probabilities.
    """
    trigram_probs = {}

    for context, counter in trigram_counts.items():
        total = sum(counter.values())
        trigram_probs[context] = {}

        for word, count in counter.items():
            trigram_probs[context][word] = count / total

    return trigram_probs


def predict_next_words(word1, word2, trigram_probs, top_n=3):
    """
    Return the top N next-word predictions.
    """
    context = (word1.lower(), word2.lower())

    if context not in trigram_probs:
        return []

    predictions = list(trigram_probs[context].items())
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:top_n]


def get_context_words(prompt):
    """
    Return the final two words from a prompt.
    Example:
    'The cat sat on the ___' -> ('on', 'the')
    """
    cleaned = prompt.replace("___", "").strip().lower()
    cleaned = cleaned.translate(str.maketrans("", "", string.punctuation))
    words = cleaned.split()

    if len(words) < 2:
        return None

    return words[-2], words[-1]


def print_attention_simulation(sentence, weights, predicted_word):
    """
    Print a simple manual attention simulation.
    """
    print("\nSimulated Attention Analysis")
    print(f"Sentence: {sentence}")
    print("Attention Weights:")

    for word, weight in weights.items():
        print(f"  {word}: {weight:.2f}")

    print(f"Transformer-style prediction: {predicted_word}")


def main():
    # Small training corpus
    text = """
    The cat sat on the mat.
    The dog sat on the rug.
    I want to eat pizza.
    I want to eat pasta.
    I want to eat dinner.
    She picked up the phone because it rang.
    She picked up the phone because she was expecting a call.
    The student opened the book because the assignment was difficult.
    The chef cooked dinner because the guests were hungry.
    """

    tokens = preprocess_text(text)
    trigram_counts = build_trigram_counts(tokens)
    trigram_probs = build_trigram_probabilities(trigram_counts)

    prompts = [
        "The cat sat on the ___",
        "I want to eat ___",
        "She picked up the phone because ___"
    ]

    print("=== TRIGRAM MODEL RESULTS ===")

    for prompt in prompts:
        context = get_context_words(prompt)

        print(f"\nPrompt: {prompt}")

        if not context:
            print("Not enough words for trigram context.")
            continue

        word1, word2 = context
        predictions = predict_next_words(word1, word2, trigram_probs, top_n=3)

        print(f"Trigram context used: ({word1}, {word2})")

        if not predictions:
            print("No trigram predictions found.")
        else:
            print("Top predictions:")
            for i, (word, prob) in enumerate(predictions, start=1):
                print(f"{i}. {word} ({prob:.3f})")

    # Example of simulated transformer-style attention analysis
    sentence = "She picked up the phone because ___"

    attention_weights = {
        "she": 0.15,
        "picked": 0.05,
        "phone": 0.35,
        "because": 0.10,
        "up": 0.05,
        "the": 0.05,
        "context of answering": 0.25
    }

    predicted_word = "it"

    print_attention_simulation(sentence, attention_weights, predicted_word)


if __name__ == "__main__":
    main()