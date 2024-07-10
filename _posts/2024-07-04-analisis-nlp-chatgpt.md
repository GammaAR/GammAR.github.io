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
In this blog, I will try to use a set of data from [kaggle](https://www.kaggle.com/datasets/ashishkumarak/chatgpt-reviews-daily-updated) to analyze its sentiment component from people's reviews called polarity (basically a standard to determine whether a comment has negative or positive meaning). It involves several steps:

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

If we put ourselves in the developer's shoes, we usually want the product we developed to become increasingly popular daily. To achieve this, we need to monitor the usage of the product daily, monthly, or yearly. In this case, we will examine the number of reviews each day from June 23, 2023, to May 24, 2024. During this period, three peaks appeared (on November 23, 2023, March 26, 2024, and May 17, 2024), which carried the most important data: people's feedback and reviews. We will analyze the time from the foot of the peak on both sides:

#### 21 November 2023 - 27 November 2023  
In this time range, the comments with a positive polarity and the most thumbs-up were mainly excited about the new voice chat mode. The [update](https://help.openai.com/en/articles/6825453-chatgpt-release-notes) itself happened on 21 November 2023 which can be interpreted as the probable cause of the first peak in reviews count. 

>The voice mode is truly revolutionary. Having a conversation with AI without having to press any buttons or wait for much delay is amazing. I can put my phone in my pocket and use my head phones and have a helpful, intelligent buddy at all times. It is not like Alexa, I can speak totally normally. My lack of a star is because it doesn't have any banter (like PI) and doesn't reflect on much of what it is saying, so that comes across as a bit cold. This is a perfect for random queries though!  

This new mode comes with the benefit of letting people ask the chatGPT effortlessly and more easily. However, the big update also comes with a problem, which is bugs. This can be explained by the negative polarity review with the most thumbs-up below,

>Not good. Once you change to voice mode, the menus disappear, so there is then no way to change it back to keyboard mode, or change any settings. You're stuck with voice and can't change anything. Another problem, you can't use voice to stop the app. The phone continues listening to you so you need to physically unlock your phone and shut down the app. If these two problems are fixed I'll change my review to 5 stars.

#### 19 March 2024 - 6 April 2024
The second peak happened between 19 March 2024 and 6 April 2024, if we only look at the top reviews provided, we can't see much information as the review itself only talks about how good it is,
>absolutely a terrific app I practically can't live without it anymore. the improvements that they've made so far so Great that I'm going to get the premium version for the $20 a month just to support the creators because of how good even the free option are. only thing is unfortunately certain features on the web app from opening the website on a mobile browser or still unavailable in the application, notably the ability to edit prior statements by the user, as well as switch between responses.

But if we look at the news story, there is one news that is very intriguing and can be a clue to what happened, its probably because of the talks between Apple and Google Gemini and OpenAI from multiple big news sources, [Bloomberg](https://www.bloomberg.com/news/articles/2024-03-18/apple-in-talks-to-license-google-gemini-for-iphone-ios-18-generative-ai-tools), [New York Times](https://www.nytimes.com/2024/03/19/technology/apple-google-ai-iphone.html), and [CNN](https://edition.cnn.com/2024/03/18/tech/apple-ai-google-gemini/index.html). The most probable explanation is, that it appears the talk has made the ChatGPT name more widely known, proven by the reviews mainly consisting of their appreciation of the application generally, but not from their newest feature. The same content was also proven from the negative polarity reviews with the most thumbs-up.

>I'm deeply disappointed with the ChatGPT app's performance. It consistently fails to follow even the simplest of instructions, making it incredibly frustrating to use. Despite clear and explicit requests, the responses provided are often irrelevant and off-topic. It's evident that the app struggles to comprehend basic commands and lacks the ability to adapt appropriately. As a user, it's incredibly frustrating to encounter such limitations, especially when seeking assistance or guidance.

#### 12 May 2024 - 24 May 2024
The last peak appeared on 17 May 2024, in this time range, OpenAI also released [the version GPT-4o](https://help.openai.com/en/articles/6825453-chatgpt-release-notes), yet the positive comment with the most thumbs-up talks about how the developer added latex rendering which is more of a feature update than a newest version update. 

>Edited my review: thank you for adding latex rendering in the app!

However, if we only talks about the newest feature only, it won't explain how ChatGPT able to reached a highest number of reviews peak at 2024. The only explanation is that the hype of GPT-4o combined with the good feature update heavily affected the number. Also, negative reviews with most thumbs-up didn't talk about the newest version, more about the newest voice update.

>Update 3: Sky's voice was slightly tweaked. But I prefer the older voice to the new one. The old one was neutral without sounding robotic. The new voice is a little expressive, but it sometimes it doesn't fit the response, especially if the response is a horror story i asked chatgpt to write. Overall, good App experience so far. The UI still needs some sort of way to visually distinguish between prompts and replies. Right now, everything looks flat with no visual differentiation.

## Next Business Strategy
From the information provided, there are a few strategies that can be implemented by the company to make the application even more successful.

### 1. User Expectations and Updates
Peaks in reviews around major updates indicate that users are highly engaged when new features are announced. This suggests that clear communication and managing expectations around new features are crucial. This also can be done by keeping the development and release of new features on a consistent schedule to maintain user interest and engagement. 

### 2. Feature Announcements 
The increase in reviews in May, linked to the announcement of GPT-4 Omni, indicates that users are very responsive to new, innovative features. This suggests that regularly introducing significant improvements can keep users engaged and attract new users. This can be done by ensuring the users always well-informed about the future updates and how to address any issues they might encounter.

### 3. Focus on Connecting with Big Names
The same can be said from the peak reviews in March, OpenAI connection to big companies such as Apple has leave major impact on the loyal Apple user and other people who basically never try to use ChatGPT App, this approach can ensure the company is widely known and more people will try to use ChatGPT.

