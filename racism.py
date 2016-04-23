## STAT 431 Mini-Project
##
## Script for Sentiment Analysis on Reddit Comments that Claim
## "Not Racist, But...", using TextBlob and Polarity Lexicons
##
## Author: Ege Sagduyu (esagduyu)
## Date: October 23rd, 2015

import numpy as np
from textblob import TextBlob
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats
import pylab
from nltk import word_tokenize

# Method 1: get a polarity value (between [-1.0,1.0]) for the entire sentence
neg_score = [TextBlob(line).sentiment.polarity for line in open('corpus.txt')]
i = 0
j = 0
# get rid of all the zero values (since we are only interested in polar opinions
for val in neg_score:
    if val == 0.0:
        i += 1
    if val < 0:
        j += 1
num_comments = len(neg_score)
## calculate the sample SD, mean and z-scores
for ctr in xrange(0,i):
    neg_score.remove(0.0)
n = np.array(neg_score)
n.sort()
std = np.std(n)
mean = np.mean(n)
print mean, std, np.median(n), np.subtract(*np.percentile(n, [75, 25])), len(n)
zscores = np.array([(x - mean) / std for x in n])
# draw the QQ plot
sm.qqplot(zscores, line='45', fit=True)
pylab.title("Normal QQ Plot (TextBlob)")
pylab.show()
# draw the histogram
plt.hist(n, normed=True)
fit = stats.norm.pdf(n, mean, std)
plt.plot(n,fit,'-o')
plt.title("Polarity Histogram for Reddit Comments (TextBlob)")
plt.xlabel("Polarity Score")
plt.ylabel("Frequency")
plt.show()

# calculate the 95% CI for this sample
conf_int = stats.norm.interval(0.95, loc=mean, scale=std/np.sqrt(len(n)))
print conf_int
print stats.ttest_1samp(n,0.0)[1]/2


"""
## This was the original method I started with to investigate what a much more simplistic
## bag-of-words approach would yield in terms of conveyed negativity through polarity lexicons,
## but I left it out to keep the focus of my article clear. However, I leave the code as you might
## be curious to see what plots and stats they convey. I also included the charts created by this section,
## which will clearly demonstrate why I chose the TextBlob method.

## Original: Calculate individual polarities by using a bag-of-words approach
# first create the polarity lexicons
# positive words list
f = open("positive-words.txt")
pos_words = set(f.read().split())
f.close()

# negative words list
f = open("negative-words.txt")
neg_words = set(f.read().split())
f.close()

# now calculate the polarity score (between [-1.0,1.0])
l = list()
f = open("corpus.txt")
for line in f:
    toks = word_tokenize(line)
    # negativity is initialized as 1 to offset one guaranteed instance of the word "racist"
    neg = 1.0
    total = 0.0
    for token in toks:
        # we do not consider words that are not in either lexicons
        if token.lower() in neg_words:
            neg -= 1
            total += 1
        if token.lower() in pos_words:
            neg += 1
            total += 1
    polarity_score = 0.0
    if total != 0.0 and neg / total != 0.0:
        polarity_score = neg / total
        l.append(polarity_score)
f.close()
neg_bow = np.array(l)
std_bow = neg_bow.std()
mean_bow = neg_bow.mean()
print mean_bow, std_bow, np.median(neg_bow), np.subtract(*np.percentile(neg_bow, [75, 25])), len(neg_bow)
zscores_bow = np.array([(x - mean_bow) / std_bow for x in neg_bow])
sm.qqplot(zscores_bow, line='45', fit=True)
pylab.title("Normal QQ Plot (Polarity Lexicons)")
pylab.show()
plt.hist(neg_bow, normed=True)
plt.title("Polarity Histogram for Reddit Comments (Polarity Lexicons)")
plt.xlabel("Polarity Score")
plt.ylabel("Frequency")
plt.show()
"""