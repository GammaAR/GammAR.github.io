---
layout: post
title: ChatGPT Application's Reviews
subtitle: NLP Analysis of ChatGPT App From Play Store
cover-img: /assets/img/Chatgpt_bg.jpg
thumbnail-img: /assets/img/chatgpt_thumbnail.jpg
share-img: /assets/img/Chatgpt_bg.jpg
tags: [Application, NLP]
author: Gamma Althafiansyah Rosyidin
---

Recently I've been curious about the ChatGPT application which has been popular for the past 2 years. ChatGPT is a virtual assistant developed by OpenAI that uses Natural Language Processing (NLP) to understand how human communication works daily. It consumes a large set of data text to learn the structure of paragraphs, sentences, words, or even the hidden meaning behind words. Currently, there are 3 different versions of ChatGPT called GPT (Generative Pre-training Transformer) which act as a brain for the ChatGPT, they are:

#### GPT-3.5
This is the default version of ChatGPT and has the lowest capabilities among the other three. This version offers basic responses but is sufficient for daily uses, the minus is that this version is also the least reliable and can't predict nor access information beyond 2022.

#### GPT-4
This version enhances the accuracy and adds the ability to access websites, it's powerful enough to be able to help your daily task with [87.2%](https://www.nature.com/articles/s41598-024-58760-x#:~:text=Compared%20to%20its%20previous%20version,(p%20%3D%200.035)%20respectively.) accuracy compared to the previous version which is GPT-3.5 that has only [47.4%](https://www.nature.com/articles/s41598-024-58760-x#:~:text=Compared%20to%20its%20previous%20version,(p%20%3D%200.035)%20respectively.) accuracy. This version brings a lot to the table but also comes with a cost of 20$ monthly subscription to OpenAI (plus user) which is a great option depending on what you do.

#### GPT-4o
This is the latest version and a better version of GPT-4. It has the same intelligence level as GPT-4 but with faster response, improved capabilities, and better at visual and audio understanding. Moreover, it supports over 50 languages, making it flexible and can be used globally. Currently, this version has limited messages per 3 hours for free user and _plus user_, the latter, however, will have a 5x greater message limit than _free user_. I have been using the free version of ChatGPT for study and revision since OpenAI has given us free usage of this very version.

## Natural Language Processing (NLP) Analysis
Natural language processing (NLP) is a subfield of computer science and artificial intelligence (AI) that uses machine learning to enable computers to understand and communicate with human language.  
In this blog, I will try to use a set of data from [kaggle](https://www.kaggle.com/datasets/ashishkumarak/chatgpt-reviews-daily-updated) to analyze its sentiment component from people's reviews. It involves several steps:

### Data Cleaning 
The data obtained must be cleaned from missing data, and incorrect data type, and free from duplicates. This optimization ensures the data will be free from bias and will reduce memory usage. We will use **sentiment analysis libraries** such as _TextBlob_ and _VADER (Valence Aware Dictionary and sEntiment Reasoner)_, in this case, we will try to focus on using VADER (with VADER, we don't need to remove emojis or numbers). With TextBlob however, we need to remove any numbers and use the _regex_ library to also [remove emojis](https://medium.com/swlh/analyzing-product-reviews-with-natural-language-processing-toolkit-nltk-b05ad87bad00) to reduce noise and increase model performance by focusing on the relevant feature which is the text itself since it cannot specifically handle emojis.  

Regex will also be used for [contraction transformations](https://www.analyticsvidhya.com/blog/2020/04/beginners-guide-exploratory-data-analysis-text-data/) which involve converting contracted forms of words (e.g., "ain't," "'s," "aren't") into their full forms (e.g., "are not," "is," "are not").

~~~
contractions_dict = { "ain't": "are not","'s":" is","aren't": "are not",
                     "can't": "cannot","can't've": "cannot have",
                     "'cause": "because","could've": "could have","couldn't": "could not",
                     "couldn't've": "could not have", "didn't": "did not","doesn't": "does not",
                     "how'd": "how did","how'd'y": "how do you","how'll": "how will",
                     "I'd": "I would", "I'd've": "I would have","I'll": "I will",
                     "I'll've": "I will have","I'm": "I am","I've": "I have", "isn't": "is not",
                     "it'd": "it would","it'd've": "it would have","it'll": "it will",
                     "it'll've": "it will have", "let's": "let us","ma'am": "madam",
                     "mayn't": "may not","might've": "might have","mightn't": "might not", 
                     "mightn't've": "might not have","must've": "must have","mustn't": "must not",
                     "mustn't've": "must not have", "needn't": "need not",
                     "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
                     "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
                     "shan't've": "shall not have","she'd": "she would","she'd've": "she would have",
                     "she'll": "she will", "she'll've": "she will have","should've": "should have",
                     "shouldn't": "should not", "shouldn't've": "should not have","so've": "so have",
                     "that'd": "that would","that'd've": "that would have", "there'd": "there would",
                     "there'd've": "there would have", "they'd": "they would",
                     "they'd've": "they would have","they'll": "they will",
                     "they'll've": "they will have", "they're": "they are","they've": "they have",
                     "to've": "to have","wasn't": "was not","we'd": "we would",
                     "we'd've": "we would have","we'll": "we will","we'll've": "we will have",
                     "we're": "we are","we've": "we have", "weren't": "were not","what'll": "what will",
                     "what'll've": "what will have","what're": "what are", "what've": "what have",
                     "when've": "when have","where'd": "where did", "where've": "where have",
                     "who'll": "who will","who'll've": "who will have","who've": "who have",
                     "why've": "why have","will've": "will have","won't": "will not",
                     "won't've": "will not have", "would've": "would have","wouldn't": "would not",
                     "wouldn't've": "would not have","y'all": "you all", "y'all'd": "you all would",
                     "y'all'd've": "you all would have","y'all're": "you all are",
                     "y'all've": "you all have", "you'd": "you would","you'd've": "you would have",
                     "you'll": "you will","you'll've": "you will have", "you're": "you are",
                     "you've": "you have"}

# Regular expression for finding contractions
contractions_re=re.compile('(%s)' % '|'.join(contractions_dict.keys()))

# Function for expanding contractions
def expand_contractions(text,contractions_dict=contractions_dict):
  def replace(match):
    return contractions_dict[match.group(0)]
  return contractions_re.sub(replace, text)

# Expanding Contractions in the reviews
df['cleaned_min']=df['cleaned_min'].apply(expand_contractions)
~~~

After cleaned, we can use VADER library as shown below,
~~~
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
df['polarity_vader'] = df['cleaned_min'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

df['polarity_vader']
~~~

The benefits VADER has over TextBlob is that in case the reviews only have emojis, VADER can handle and score the polarity properly, it is also suitable for social media comment analysis.  
Then we save the data obtained in xlsx format.
~~~
df.to_excel("df_clean_new.xlsx", index=False)
~~~

### Sentiment Data Analysis
Finally, the clean data obtained will be analyzed using Tableau Public for great data visualization.

#### Reviews Count
<p align="center">
  <img src="https://raw.githubusercontent.com/GammaAR/GammaAR.github.io/master/assets/img/chatgpt/Reviews%20Count%20ChatGPT.png" alt="New Cat" width="900" height="500">
</p>

If we see the graph above,



