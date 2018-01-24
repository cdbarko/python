from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag

from nltk.classify import NaiveBayesClassifier
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import extract_unigram_feats, mark_negation
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.metrics import accuracy_score

from unidecode import unidecode
import pandas as pd
import random

data = pd.read_csv(r'''C:\Users\cindy_000\Downloads\labeledTrainData.tsv\labeledTrainData.tsv''', header=0, delimiter='\t', quoting=3)

# 25000 movie reviews
print(data.shape) # (25000, 3)
print(data["review"][0]) # Check out the review
print(data["sentiment"][0]) # Check out the sentiment (0/1)

#split for training and testing

sentiment_data = list(zip(data["review"], data["sentiment"]))
random.shuffle(sentiment_data)

# 80% for training
train_X, train_y = list(zip(*sentiment_data[:20000]))

# Keep 20% for testing
test_X, test_y = list(zip(*sentiment_data[20000:]))

print(train_X[0])
print(train_y[0])


lemmatizer = WordNetLemmatizer()
 
 
def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None
 
 
def clean_text(text):
    text = text.replace("<br />", " ")
    #text = text.decode("utf-8")
 
    return text
 
 
def swn_polarity(text):
    """
    Return a sentiment polarity: 0 = negative, 1 = positive
    """
 
    sentiment = 0.0
    tokens_count = 0
 
    text = clean_text(text)
 
 
    raw_sentences = sent_tokenize(text)
    for raw_sentence in raw_sentences:
        tagged_sentence = pos_tag(word_tokenize(raw_sentence))
 
        for word, tag in tagged_sentence:
            wn_tag = penn_to_wn(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue
 
            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if not lemma:
                continue
 
            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                continue
 
            # Take the first sense, the most common
            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())
 
            sentiment += swn_synset.pos_score() - swn_synset.neg_score()
            tokens_count += 1
 
    # judgment call ? Default to positive or negative
    if not tokens_count:
        return 0
 
    # sum greater than 0 => positive sentiment
    if sentiment >= 0:
        return 1
 
    # negative sentiment
    return 0
 
 
# since we're shuffling, you'll get diffrent results
print(swn_polarity(train_X[0]), train_y[0]) # 1 1
print(swn_polarity(train_X[1]), train_y[1]) # 0 0
print(swn_polarity(train_X[2]), train_y[2]) # 0 1
print(swn_polarity(train_X[3]), train_y[3]) # 1 1
print(swn_polarity(train_X[4]), train_y[4]) # 1 1


# compute accuracy of the SWN method

pred_y = [swn_polarity(text) for text in test_X]
 
print("Accuracy of SentiWordnet: ", accuracy_score(test_y, pred_y)) # 0.6518


 
# mark_negation appends a "_NEG" to words after a negation untill a punctuation mark.
# this means that the same after a negation will be handled differently 
# than the word that's not after a negation by the classifier

print(mark_negation("I like the movie .".split()))        # ['I', 'like', 'the', 'movie.']
print(mark_negation("I don't like the movie .".split()))  # ['I', "don't", 'like_NEG', 'the_NEG', 'movie._NEG']
 
# The nltk classifier won't be able to handle the whole training set
TRAINING_COUNT = 5000
 
analyzer = SentimentAnalyzer()
vocabulary = analyzer.all_words([mark_negation(word_tokenize(unidecode(clean_text(instance)))) 
                                 for instance in train_X[:TRAINING_COUNT]])
print("Vocabulary: ", len(vocabulary)) # 1356908
 
print("Computing Unigran Features ...")
unigram_features = analyzer.unigram_word_feats(vocabulary, min_freq=10)

print("Unigram Features: ", len(unigram_features)) # 8237
 
analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_features)
 
# Build the training set
_train_X = analyzer.apply_features([mark_negation(word_tokenize(unidecode(clean_text(instance)))) 
                                    for instance in train_X[:TRAINING_COUNT]], labeled=False)
 
# Build the test set
_test_X = analyzer.apply_features([mark_negation(word_tokenize(unidecode(clean_text(instance)))) 
                                   for instance in test_X], labeled=False)
 
trainer = NaiveBayesClassifier.train
classifier = analyzer.train(trainer, zip(_train_X, train_y[:TRAINING_COUNT]))
 
score = analyzer.evaluate(zip(_test_X, test_y))
print("Accuracy of SentimentAnalyzer: ", score['Accuracy']) # 0.8064 for TRAINING_COUNT=5000


 
vader = SentimentIntensityAnalyzer()

def vader_polarity(text):
    """ Transform the output to a binary 0/1 result """
    score = vader.polarity_scores(text)
    return 1 if score['pos'] > score['neg'] else 0
 
print(vader_polarity(train_X[0]), train_y[0]) # 0 1
print(vader_polarity(train_X[1]), train_y[1]) # 0 0
print(vader_polarity(train_X[2]), train_y[2]) # 1 1
print(vader_polarity(train_X[3]), train_y[3]) # 0 1
print(vader_polarity(train_X[4]), train_y[4]) # 0 0
 
pred_y = [vader_polarity(text) for text in test_X]
print("Accuracy of VADER SentimentIntensityAnalyzer: ", accuracy_score(test_y, pred_y)) # 0.6892
