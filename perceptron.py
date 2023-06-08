import random
from collections import defaultdict
from typing import Any

Token = tuple[str, str]  # (word, tag)
TaggedSentence = list[Token]


class PerceptronPOSTagger:
    """
    A multi-class perceptron for POS-tagging.
    """

    def __init__(self):
        # Maps feature -> tag -> weights.
        self.weights = defaultdict(lambda: defaultdict(float))

        # The set of tags this tagger uses.
        self.tags = set()  # type: set[str]

    def tag(self, words: list[str]) -> TaggedSentence:
        """
        Tag each word in a sentence.

        :param words: the words in the sentence.
        :return: a tagged sentence: a list of (word, tag) tuples.
        """
        predicted_tags = []
        for word_index, word in enumerate(words):
            features = self.get_features(word, word_index, words, predicted_tags)
            predicted_tag = self.predict(features)
            predicted_tags.append(predicted_tag)

        return list(zip(words, predicted_tags))

    def predict(self, features: list[Any]) -> str:
        """
        Predict a tag from a list of features.

        :param features: a list of features.
        :return: the predicted tag.
        """

        # The computed the scores for each tag.
        scores = defaultdict(float)

        # For each feature, sum the weight for each tag to the score.
        for feature in features:
            if feature in self.weights:

                weights = self.weights[feature]
                for tag, weight in weights.items():
                    scores[tag] += weight

        # Return the tag with the highest score.
        return max(self.tags, key=lambda tag: scores[tag])

    def train(self, sentences: list[TaggedSentence], iterations: int = 1):
        """
        Train the perceptron using gold-tagged sentences.

        :param sentences: a list of tagged sentences, where a sentence is a list of
        (word, tag) tuples.
        :param iterations: the number of times the tagger will be trained on the data.
        """

        # Maps feature -> tag -> weights.
        self.weights = defaultdict(lambda: defaultdict(float))
        self.tags = set()  # type: set[str]

        # The accumulated adjustments for each feature-weight.
        total_adjustments = defaultdict(lambda: defaultdict(float))
        iter_counter = 1

        # Adjust the weight for the given feature and tag.
        def adjust_weight(feature, tag, increment):
            self.weights[feature][tag] += increment
            total_adjustments[feature][tag] += iter_counter * increment

        # Iteratively train the tagger.
        for iteration in range(iterations):

            # Train on each sentence.
            for sentence in sentences:
                previous_predicted_tags = []
                words = [word for word, tag in sentence]

                # Train on each word in sentence.
                for word_index, (word, gold_tag) in enumerate(sentence):
                    self.tags.add(gold_tag)

                    # Extract the features.
                    features = self.get_features(word, word_index, words,
                                                 previous_predicted_tags)

                    # Make a prediction.
                    predicted_tag = self.predict(features)
                    previous_predicted_tags.append(predicted_tag)

                    # Adjust relevant feature-weights if prediction failed.
                    if predicted_tag != gold_tag:
                        for feature in features:
                            adjust_weight(feature, gold_tag, 1)
                            adjust_weight(feature, predicted_tag, -1)

                    iter_counter += 1

            random.shuffle(sentences)

        # Average the weights, because later adjustments affect the weights
        # more than earlier.
        for feature, weights in self.weights.items():
            for tag in weights:
                self.weights[feature][tag] -= total_adjustments[feature][tag] / iter_counter

    def get_features(self, word: str, word_index: int, words: list[str],
                     previous_predicted_tags: list[str]) -> list[Any]:
        """
        Extracts the features from a given context.

        :param word: the current word to be tagged.
        :param word_index: the index of the current word.
        :param words: the words in the sentence of the current word.
        :param previous_predicted_tags: the previously predicted tags in the sentence.

        :return: a list of features.
        """

        features = []

        def add(feature):
            features.append((len(features), feature))

        # Current and surrounding words.
        add(word)
        add(words[word_index - 1] if word_index > 0 else '!BOS')
        add(words[word_index + 1] if word_index + 1 < len(words) else '!EOS')

        # Previous tag.
        add(previous_predicted_tags[word_index - 1] if word_index > 0 else '!BOS')

        # Prefixes and suffixes of current word.
        add(word[-4:])
        add(word[:4])
        add(word[-2:])
        add(word[:2])

        return features


# Testing and evaluation ------------


def tagged_sentences(file_name) -> list[TaggedSentence]:
    """
    Load a list of tagged sentences from a file.

    In the file, each line consists of a word and tag pair, which are separated by a tab
    character. Empty lines indicate end of sentence. In other words, a sentence is

    :param file_name: the name of the file.
    :return: a list of tagged sentences, where a sentence is a list of (word, tag) tuples.
    """
    sentences = []
    with open(file_name) as source:

        tagged_sentence = []
        for line in source:

            # Newline indicates end of sentence, -> yield sentence.
            if line == '\n':
                sentences.append(tagged_sentence)
                tagged_sentence = []
            else:
                fields = line.split('\t')
                word = fields[0].strip().lower()
                tag = fields[1]
                tagged_sentence.append((word, tag))

    return sentences


def evaluate_tagger(tagger: PerceptronPOSTagger, test_sentences: list[TaggedSentence]):
    """
    Compute the accuracy of the tagger given a list of test sentences. This computes
    two types of accuracies: 1) the accuracy of all tags, and 2) the accuracy of
    all correctly tagged sentences (that is, the entire sentence is correctly tagged).

    :param tagger: the perceptron tagger.
    :param test_sentences: a list of tagged sentences, where a sentence is a list of
    (word, tag) tuples.
    """

    correct_tags = 0
    total_tags = 0
    correct_sentences = 0
    total_sentences = 0

    # Tag and evaluate each sentence.
    for sentence in test_sentences:
        words = [word for word, _ in sentence]
        gold_tags = [tag for _, tag in sentence]

        # Tag words.
        predicted_tags = [tag for _, tag in tagger.tag(words)]

        # Count number of correct tags.
        correct_in_sentence = 0
        for predicted_tag, gold_tag in zip(predicted_tags, gold_tags):
            if predicted_tag == gold_tag:
                correct_in_sentence += 1

        # Sum up with the global statistics.
        total_tags += len(predicted_tags)
        correct_tags += correct_in_sentence
        total_sentences += 1

        # Check if the entire sentence was correctly tagged.
        if correct_in_sentence == len(predicted_tags):
            correct_sentences += 1

    print('Accuracy tags: ', (correct_tags / total_tags))
    print('Accuracy sentences: ', (correct_sentences / total_sentences))


def test():
    """
    Test the perceptron tagger class, by first training it, and then testing
    it on a dev-set and a test-set. The dev-set is used for tweaking the features.
    """
    tagger = PerceptronPOSTagger()

    # Train tagger.
    print("Starting training.")
    train_sentences = tagged_sentences('suc3-pos-train.txt')
    tagger.train(train_sentences, iterations=1)
    print("Training complete.")

    # Test tagger on dev set.
    print('Dev:')
    dev_sentences = tagged_sentences('suc3-pos-dev.txt')
    evaluate_tagger(tagger, dev_sentences)

    # Test tagger on test set.
    print('Test:')
    test_sentences = tagged_sentences('suc3-pos-test.txt')
    evaluate_tagger(tagger, test_sentences)


if __name__ == '__main__':
    test()
