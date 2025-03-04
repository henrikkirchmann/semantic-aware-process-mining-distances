import nltk
import numpy as np
import collections
from nltk.corpus import gutenberg, europarl_raw, reuters, udhr
from nltk.tokenize import TweetTokenizer

# Ensure necessary NLTK data is available
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("gutenberg")
nltk.download("europarl_raw")
nltk.download("reuters")
nltk.download("udhr")

def get_udhr_sentences(lang_code):
    """ Returns UDHR sentences if available, otherwise None """
    try:
        return udhr.sents(lang_code)
    except OSError:
        print(f"Warning: UDHR corpus for {lang_code} not found. Skipping...")
        return None

LANGUAGES = {
    "English": gutenberg.sents("austen-emma.txt"),
    "Spanish": europarl_raw.spanish.sents(),
    "French": europarl_raw.french.sents(),
    "German": europarl_raw.german.sents(),
    "Italian": europarl_raw.italian.sents(),
    "Dutch": europarl_raw.dutch.sents(),
    "Chinese": get_udhr_sentences("Chinese-UTF8"),  # Try another available option
    "Japanese": get_udhr_sentences("ja"),
    "Korean": get_udhr_sentences("ko"),
    "Persian": get_udhr_sentences("fa"),
}


# Function to preprocess sentences
def preprocess_sentences(sentences):
    """
    Tokenizes and cleans sentences, removing punctuation while keeping sentence structure.

    :param sentences: List of tokenized raw sentences.
    :return: List of tokenized sentences without punctuation.
    """
    tokenizer = TweetTokenizer()

    tokenized_sentences = [
        [word for word in tokenizer.tokenize(" ".join(sentence)) if word.isalpha()]
        for sentence in sentences
    ]

    return tokenized_sentences


# Function to compute relative positions
def compute_relative_positions(sentences):
    """
    Computes the average relative position, standard deviation, and probability of the top 10 words.

    :param sentences: List of tokenized sentences.
    :return: Dictionary with words as keys and (mean relative position, std deviation, probability) as values.
    """
    if not sentences:
        return {}

    all_words = [word for sentence in sentences for word in sentence]
    word_counts = collections.Counter(all_words)
    total_words = sum(word_counts.values())

    # Get the top 10 most frequent words
    top_words = [word for word, _ in word_counts.most_common(10)]

    # Store relative positions of top words
    word_positions = {word: [] for word in top_words}
    word_probabilities = {word: word_counts[word] / total_words for word in top_words}

    for sentence in sentences:
        for pos, word in enumerate(sentence):
            if word in top_words:
                relative_pos = pos / len(sentence)  # Relative position within the sentence
                word_positions[word].append(relative_pos)

    # Compute mean and standard deviation
    result = {
        word: (np.mean(pos_list), np.std(pos_list), word_probabilities[word])
        for word, pos_list in word_positions.items()
    }

    return result


# Run analysis for each language
language_statistics = {}

for language, corpus_data in LANGUAGES.items():
    print(f"Processing {language} dataset...")

    try:
        tokenized_sentences = preprocess_sentences(corpus_data)

        if tokenized_sentences:
            language_statistics[language] = compute_relative_positions(tokenized_sentences)
        else:
            print(f"Skipping {language} due to missing or empty corpus.")

    except Exception as e:
        print(f"Error processing {language}: {e}")
        continue

# Print results
for language, stats in language_statistics.items():
    print(f"\nLanguage: {language}")
    for word, (mean_pos, std_dev, probability) in stats.items():
        print(f"  Word: {word}, Mean Position: {mean_pos:.4f}, Std Dev: {std_dev:.4f}, Probability: {probability:.4f}")
