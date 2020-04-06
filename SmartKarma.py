#!/usr/bin/env python
# coding: utf-8

# cd stanford-corenlp-full-2018-10-05
# java -mx5g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 10000
from textblob import TextBlob
from pycorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost:9000')


# In[7]:


# Neutral (2) and Negative (1), the range is from Very Negative (0) to Very Positive (4)
def sentiment_stanford(text):
    ls = []
    res = nlp.annotate(text, properties={
                       'annotators': 'sentiment',
                       'outputFormat': 'json',
                       'timeout': 1000})
    for s in res["sentences"]:
        ls.append((" ".join([t["word"] for t in s["tokens"]]), s["sentimentValue"]))
    return ls


# In[3]:


# Polarity is a float value within the range [-1.0 to 1.0] where 0 indicates neutral,
# +1 indicates a very positive sentiment and -1 represents a very negative sentiment

# Subjectivity is a float value within the range [0.0 to 1.0] where 0.0 is very objective and 1.0 is very subjective
def sentiment_textblob(text):
  blob = TextBlob(text)
  return blob.sentiment


# In[5]:


def sentiment(text):
    res = nlp.annotate(text, properties={
                       'annotators': 'sentiment',
                       'outputFormat': 'json',
                       'timeout': 1000})

    for s in res["sentences"]:
        token = " ".join([t["word"] for t in s["tokens"]])
        print(sentiment_stanford(token))
        print(sentiment_textblob(token))

sentiment("coronavirus has caused the stock market to crash\n things are looking good for the tech industry\n trump's impeachment is going well\n")
