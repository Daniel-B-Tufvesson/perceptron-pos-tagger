# Multi-class Perceptron POS Tagger
This is a multi-class perceptron for part-of-speech (POS) tagging Swedish words in sentences. POS-tagging is the automatic assignment of part-of-speech tags (i.e. word classes) to words in a sentence. 

This tagger is a reimplementation of the tagger by Honnibal (2013). The purpose of reimplementing it was for me to gain an understanding of how multi-class perceptrons work.


## Running the tagger
Running the script trains the tagger on the test data (`suc3-pos-train.txt`), and then evaluates it on the dev data (`suc3-pos-dev.txt`) and the test data (`suc3-pos-test.txt`).

## How multi-class perceptrons work
A multi-class perceptron uses several perceptrons in order to classify data as belonging to one of several classes (in this case the classes are POS-tags, and the data is the specific word to be tagged). This is in contrast to the single perceptron, which can only make binary classifications, that is, classify something as either belonging or not belonging to a class. In the multi-class perceptron there is a perceptron for each class. When classifying a word, each perceptron computes a score for that word. The higher the score, the more likely it belongs to the tag of that perceptron. These scores are then compared with an argmax function which selects the tag with the highest score. This selected tag is the predicted tag of the multi-class perceptron. 

A multi-class perceptron does not have to look at only the current word to be tagged, but may look at the so-called context of the word. What the perceptron looks at are referred to as features. For POS-tagging, these features can include, but is not limited to, the previous word, the next word, the previously predicted tag, as well as prefixes and suffixes of the context words. See Figure 1 for a simple example of a multi-class perceptron tagger with three features.
![POS Perceptron.png](POS%20Perceptron.png)
Figure 1. An oversimplified multi-class perceptron which can tag a specific word a word either as a NN (noun), VB (verb), or PP (preposition). It uses only three features: the specific current word, the specific previous word, and the specific next word.

## Training and evaluation
This tagger is trained on SUCX 3.0 (Språkbanken Text, 2022), a Swedish corpus of scrambled sentences, consisting of 1,166,593 tokens and 74,245 sentences. The original corpus is an XML file consisting of a multitude of different linguistic annotation. For this project, however, only the words and their POS-tags were relevant. Therefore, I extracted these into separate files for faster reading and less data storage. This resulted in three files: 1) a training data file consisting of roughly 60% of the sentences, 2) a development data file (20%) for finding and tweaking the features, and 3) a test data file (20%) for final evaluation of the tagger. The only pre-processing done was lower-casing all words.

The accuracy for a POS tagger can be determined either on how many tags were correctly tagged (tag accuracy), or on how many sentences were correctly tagged (sentence accuracy). This tagger, with the SUCX 3.0 corpus, and trained 10 iterations, achieves the following accuracies. 

|                   | Dev data | Test data |
|-------------------|----------|-----------|
| Tag accuracy      | 96.56%   | 96.64%    |
| Sentence accuracy | 65.5%    | 65.33%    |


## Differences from Honnibal's tagger
This tagger differs from that of Honnibal's (2013) in a number of ways. Firstly, it does not rely on a tag dictionary for highly common words. According to Honnibal, a tag dictionary helps the tagger disambiguate common words. I have however not tested if this improves the accuracy for my tagger. Futhermore, I wanted a pure perceptron tagger, so relying on a tag dictionary would defeat that purpose.

Secondly, this tagger uses fewer features — 8 features compared to 14. Some features may be language dependent, so it may be the case that for Swedish, compared to English, the number of features can be reduced.

Thirdly, this implementation has everything condensed into a single class, while Honnibal's is separated into at least two classes (as far as I can see), one class for the perceptron, and another for the tagger. This is of course a design decision which depends on the purpose of the code. Honnibal's tagger is part of a framework, so code flexibility and reuse become more important, while my tagger is standalone and only for self-educational purposes, so brevity and specificity is more relevant. 

## References
Honnibal, M. (2013). A Good Part-of-Speech Tagger in about 200 Lines of Python. https://explosion.ai/blog/part-of-speech-pos-tagger-in-python

Språkbanken Text. (2022). SUCX 3.0 https://spraakbanken.gu.se/en/resources/sucx3