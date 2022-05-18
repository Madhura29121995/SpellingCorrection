from decimal import Decimal
from http.server import ThreadingHTTPServer
from lib2to3.pgen2 import token

eps = 0.0001

class UnsmoothedUnigramLM:
    def __init__(self, fname):
        self.freqs = {}
        dict = {}
        for line in open(fname):
            tokens = line.split()
            for index in range(len(tokens)):
                self.freqs[tokens[index]] = self.freqs.get(tokens[index], 0) + 1
                
        # Computing this sum once in the constructor, instead of every
        # time it's needed in log_prob, speeds things up
        self.num_tokens = sum(self.freqs.values())
        self.num_types = len(self.freqs)

    def Uni_cal_probs(self, word):
        # Compute probabilities in log space to avoid underflow errors
        # (This is not actually a problem for this language model, but
        # it can become an issue when we multiply together many
        # probabilities)
        if word in self.freqs:
            return (self.freqs[word] + 1) / (self.num_tokens + self.num_types)
        else:
            # This is a bit of a hack to get a float with the value of
            # minus infinity for words that have probability 0
            return float("-inf")

    def in_vocab(self, word):
        return word in self.freqs

    def uni_check_probs(self):
        # Hint: Writing code to check whether the probabilities you
        # have computed form a valid probability distribution is very
        # helpful, particularly when you start incorporating smoothing
        # (or interpolation). It can be a bit slow, however,
        # especially for bigram language models, so you might want to
        # turn these checks off once you're convinced things are
        # working correctly.

        # Make sure the probability for each word is between 0 and 1
        for w in self.freqs:
            assert 0 - eps < Decimal(self.Uni_cal_probs(w)) < 1 + eps
        # Make sure that the sum of probabilities for all words is 1
        assert 1 - eps < \
            sum([Decimal(self.Uni_cal_probs(w)) for w in self.freqs]) < \
            1 + eps

class UnsmoothedBigramLM:
    def __init__(self, fname):
        self.freqs = {}
        self.dict = {}
        for line in open(fname):
            tokens = line.split()
            for index in range(len(tokens)):
                if index+1<len(tokens):
                    self.dict[tokens[index] + " " + tokens[index + 1]] = self.dict.get(tokens[index] + " " + tokens[index + 1]) + 1 if self.dict.get(tokens[index] + " " + tokens[index + 1])!=None else 1
                self.freqs[tokens[index]] = self.freqs.get(tokens[index], 0) + 1
        # Computing this sum once in the constructor, instead of every
        # time it's needed in log_prob, speeds things up
        self.num_tokens = sum(self.freqs.values())
        self.num_types = len(self.freqs)

    def Bi_cal_probs(self, word, leftWord, rightWord):
        # Compute probabilities in log space to avoid underflow errors
        # (This is not actually a problem for this language model, but
        # it can become an issue when we multiply together many
        # probabilities)
        #if word in self.freqs:
        if self.dict.get(leftWord + " " + word) == None:
            self.dict[leftWord + " " + word] = 0
        if self.dict.get(word + " " + rightWord) == None:
            self.dict[word + " " + rightWord] = 0
            
        return (((self.dict.get(leftWord + " " + word) + 1) / (self.freqs[word] + self.num_types)) * ((self.dict.get(word + " " + rightWord) + 1) / (self.freqs[word] + self.num_types)))
        #else:
            # This is a bit of a hack to get a float with the value of
            # minus infinity for words that have probability 0
            #return float("-inf")

    def in_vocab(self, word):
        return word in self.freqs

class UnsmoothedInterLM:
    def __init__(self, fname):
        self.freqs = {}
        self.dict = {}
        for line in open(fname):
            tokens = line.split()
            for index in range(len(tokens)):
                if index+1<len(tokens):
                    self.dict[tokens[index] + " " + tokens[index + 1]] = self.dict.get(tokens[index] + " " + tokens[index + 1]) + 1 if self.dict.get(tokens[index] + " " + tokens[index + 1])!=None else 1
                self.freqs[tokens[index]] = self.freqs.get(tokens[index], 0) + 1
        # Computing this sum once in the constructor, instead of every
        # time it's needed in log_prob, speeds things up
        self.num_tokens = sum(self.freqs.values())
        self.num_types = len(self.freqs)

    def Inter_cal_probs(self, word, leftWord, rightWord):
        # Compute probabilities in log space to avoid underflow errors
        # (This is not actually a problem for this language model, but
        # it can become an issue when we multiply together many
        # probabilities)
        if word in self.freqs:
            if self.dict.get(leftWord + " " + word) == None:
                self.dict[leftWord + " " + word] = 0
            if self.dict.get(word + " " + rightWord) == None:
                self.dict[word + " " + rightWord] = 0

            if rightWord in self.freqs:
                return (((0.4) * self.dict.get(leftWord + " " + word) / self.freqs[word]) + ((0.4) * ((self.freqs[word] +1) / (self.num_tokens+ self.num_types)))) * ((0.4)* (self.dict.get(word + " " + rightWord) / self.freqs[word]) + ((self.freqs[rightWord]+1) / (self.num_tokens+self.num_types)))
       
        #else:
            # This is a bit of a hack to get a float with the value of
            # minus infinity for words that have probability 0
            #return float("-inf")

    def in_vocab(self, word):
        return word in self.freqs

    def Inter_check_probs(self):
        # Hint: Writing code to check whether the probabilities you
        # have computed form a valid probability distribution is very
        # helpful, particularly when you start incorporating smoothing
        # (or interpolation). It can be a bit slow, however,
        # especially for bigram language models, so you might want to
        # turn these checks off once you're convinced things are
        # working correctly.

        # Make sure the probability for each word is between 0 and 1
        for w in self.freqs:
            assert 0 - eps < Decimal(self.Inter_cal_probs(w)) < 1 + eps
        # Make sure that the sum of probabilities for all words is 1
        assert 1 - eps < \
            sum([Decimal(self.Inter_cal_probs(w)) for w in self.freqs]) < \
            1 + eps

def insertion_edits(w):
    # Return the set of strings that can be formed
    result = set()
    for i in range(len(w)+1):
       for cha in range(97,123):
           #insertion
           result.add(w[:i] + chr(cha) + w[i:])
    return result

def deletion_edits(w):
    # Return the set of strings that can be formed
    result = set()
    for i in range(len(w)):
       #deletion
       result.add(w[:i] + w[i+1:])
    return result

def substitution_edits(w):
    # Return the set of strings that can be formed
    result = set()
    for i in range(len(w)):
       for cha in range(97,123):
           #substitution
           result.add(w[:i] + chr(cha) + w[i+1:])
    return result

def transposition_edits(w):
    # Return the set of strings that can be formed
    result = set()
    for i in range(len(w)):
       #transposition
       if i+2 < len(w):
           result.add(w[:i] + w[i+1] + w[i] + w[i+2:])
       elif i+1 < len(w):
           result.add(w[:i] + w[i+1] + w[i])
    return result

if __name__ == '__main__':
    import sys

    # Look for the training corpus in the current directory
    train_corpus = 'corpus.txt' 

    # n will be '1', '2' or 'interp' (but this starter code ignores
    # this)
    n = 3

    # The collection of sentences to make predictions for
    predict_corpus = 'dev.txt'

    # Train the language model
    if (n == 1):
        lm = UnsmoothedUnigramLM(train_corpus)
    if (n == 2):
        lm = UnsmoothedBigramLM(train_corpus)
    if (n == 3):
        lm = UnsmoothedInterLM(train_corpus)

    # You can comment this out to run faster...
    #if (n == 1):
       ###lm.Bi_check_probs()
    ##lm.Inter_check_probs()

    for line in open(predict_corpus):
        # Split the line on a tab; get the target word to correct and
        # the sentence it's in
        target_index,sentence = line.split('\t')
        target_index = int(target_index)
        sentence = sentence.split()
        target_word = sentence[target_index]
        left_word = sentence[target_index - 1]
        right_word = sentence[target_index + 1]
        # Get the in-vocabulary candidates (this starter code only
        # considers deletions)
        #candidates = delete_edits(target_word)
        candidates = set.union(insertion_edits(target_word),substitution_edits(target_word),transposition_edits(target_word), deletion_edits(target_word)) 
        iv_candidates = [c for c in candidates if lm.in_vocab(c)]
        
        # Find the candidate correction with the highest probability;
        # if no candidate has non-zero probability, or there are no
        # candidates, give up and output the original target word as
        # the correction.
        best_prob = float("-inf")
        best_correction = target_word
        if (n==1):
            for ivc in iv_candidates:
                ivc_log_prob = lm.Uni_cal_probs(ivc)
                if ivc_log_prob > best_prob:
                    best_prob = ivc_log_prob
                    best_correction = ivc
        
        if (n==2):
            for ivc in iv_candidates:
                ivc_log_prob = lm.Bi_cal_probs(ivc, left_word, right_word)
                if ivc_log_prob > best_prob:
                    best_prob = ivc_log_prob
                    best_correction = ivc

        if (n==3):
            for ivc in iv_candidates:
                ivc_log_prob = lm.Inter_cal_probs(ivc, left_word, right_word)
                if ivc_log_prob != None:
                    if ivc_log_prob > best_prob:
                        best_prob = ivc_log_prob
                        best_correction = ivc
        print(best_correction)
