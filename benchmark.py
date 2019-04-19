#For Crimson Hexagon Extract Data

# 1. Open the Data file and delete the first row in Excel
# 2. Convert the type of column Date (EST) to Short Date in Excel
# 3. Save the data to csv
# 4. Clear all file from data folder
# 5. Paste the CSV file to the data folder
# 6. Run this file
# 7. See the result in plot folder

import pandas as pd
import csv
import os
import re       # regular expressions for cleaning tweets
from textblob import TextBlob         #helping with with sentiment analysis
from sklearn.feature_extraction.text import CountVectorizer
import spacy
# from spacy.lang.en import English
import nltk
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

nlp = spacy.load('en_core_web_sm')
currentDT = ""

PATH_DATA = 'data'
PATH_RESULT = 'result'

TO_INCLUDE = False
TOPIC = ['lag']

TO_EXCLUDE = False
EXCLUDE_TOPIC = ['google', 'stadia']

PLOT_NOTE1 = ""
PLOT_NOTE2 = ""

if TO_INCLUDE:
	PLOT_NOTE1 = "Include word : "+str(TOPIC)
if TO_EXCLUDE:
	PLOT_NOTE2 = "Exclude word : "+str(EXCLUDE_TOPIC)

PLOT_NOTE = PLOT_NOTE1 + " " + PLOT_NOTE2

 

# nltk.download('wordnet')

#Get the stop words
# nltk.download('stopwords')
# en_stop = set(nltk.corpus.stopwords.words('english'))


# parser = English()

def clean_text(text): 
	return ''.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())

def tokenize(text):
	# print(text)
	lda_tokens = []
	# text = clean_tweet(text)
	doc = nlp(text)
	# wordList = []
    # for i in tokens:
    #     i = clean(i)
    #     wordList += i.lower().split()
	# print(doc)
	for token in doc:
		if '\n' in token.text:
			continue
		elif '-' in token.text:
			continue
		elif ':' in token.text:
			continue
		elif token.text.isspace():
			continue
		elif token.like_url:
			lda_tokens.append('URL')
		elif token.text.startswith('@'):
			lda_tokens.append('SCREEN_NAME')
		else:
			lda_tokens.append(clean_text(token.lower_))
	return lda_tokens
   
def vectorizer(tweets):
	cv = CountVectorizer(binary=True)
	cv.fit(tweets)
	tweets = cv.transform(tweets)
	return tweets

def datetime_extraction(text):
	date = text[0:11]
	time = text[12:]
	return date, time
    
def get_sentiment(tweet):
	# create TextBlob object of passed tweet text
	result = ""
	# tweet = list(tweet)

	# print(tweet)

	# if type(tweet) != list:
	# 	result = 'unknown'
	# 	# print(result)
	# 	return result

	tw_string = ' '.join(tweet)

	# print(tw_string)

	if type(tw_string) != str:
		result = 'unknown'
		# print(result)
		return result
	analysis = TextBlob(tw_string)
	# set sentiment 
	if analysis.sentiment.polarity > 0.3:
		result = 'positive'
	elif analysis.sentiment.polarity < 0:
		result = 'negative'
	else: 
		result = 'neutral'

	# print(result)
	return result

def benchmark(tweets):

	currentDT = f"{datetime.now():%Y-%m-%d-%H-%M-%S}"
	
	tweets = tweets.loc[:,['Date (EST)','Contents','Category', 'Post Title']]

	tweets['Category'] = tweets['Category'].str.lower()

	print(tweets[0:6])

	for idx in tweets.index:
		tweets.loc[idx,'Contents'] = str(tweets.loc[idx,'Contents']) + " " + str(tweets.loc[idx,'Post Title'])

	tweets.loc[:,'Date_Ext'] = tweets.loc[:,'Date (EST)'].str.slice(0,11)
	# tweets.loc[:,'Time_Ext'] = tweets.loc[:,'Date (EST)'].str[12,]

	print(tweets.loc[0:6,'Date_Ext'])
	# tweets['token_text'] = [tokenize(item['Full Text']) for item in tweets]
	# 12:

	tweets.drop_duplicates(inplace=True)

	# print(tweets['Domain'].value_counts())

	# domain_count = tweets.groupby(['Date_Ext', 'Domain']).size().unstack(fill_value=0)

	# domain_count.to_csv(os.path.join(PATH_RESULT,currentDT+'_domain_count.csv'), index=True)
	#Plot
	# plot(domain_count['Date_Ext'], domain_count['positive'], domain_count['negative'], plot_result['neutral'], plot_result['unknown'], 'overall')
	
	#Get Only Topics
	if TO_INCLUDE:
		for ind in tweets.index:
			if any(word in str(tweets.loc[ind,'Contents']).lower() for word in TOPIC):
				continue
			else:
				tweets.drop(ind, inplace=True)

	if TO_EXCLUDE:
		for ind in tweets.index:
			if any(word in str(tweets.loc[ind,'Contents']).lower() for word in EXCLUDE_TOPIC):
				tweets.drop(ind, inplace=True)
			else:
				continue
		

	# for ind in tweets.index:
	#     #Separate each line to text_data list
	# 	# print(len(str(tweets.loc[ind,'Twitter Retweet of'])))
	# 	if 5 > len(str(tweets.loc[ind,'Twitter Retweet of'])) or "RT" == str(tweets.loc[ind,'Full Text']):
	# 		tweets.loc[ind,'retweet'] = 'retweet'
	# 	else:
	# 		tweets.loc[ind,'retweet'] = 'not_retweet'

	# retweet_count = tweets.groupby(['Date_Ext', 'retweet']).size().unstack(fill_value=0)
	
	# print(retweet_count[0:6])

	# retweet_count.reset_index(inplace=True)
	# # Plot
	# plot(retweet_count['Date_Ext'], retweet_count['retweet'], retweet_count['not_retweet'], 'retweet')

	#Tokenizing
	print("--Tokenizing--")
	result_token = []
	# tweets['token_text'] = [tokenize(item['Full Text']) for item in tweets]
	for ind in tweets.index:
	    #Separate each line to token list
	    text = tweets.loc[ind,'Contents']
	    if pd.isnull(text):
	    	result_token.append([])
	    elif (str(text).strip() != ""):
	    	# print(text)
	    	token = tokenize(tweets.loc[ind,'Contents'])
	    	# print(token)
	    	result_token.append(token)
	    else:
	    	result_token.append([])
	    # tweets.at[i,'token_text'] = token
	# print(result_token[0:6])

	#Get only specific topic
	

	tweets['token_text'] = result_token

	# tweets.set_index(['Date (EST)'],inplace=True)

	print(tweets[0:6])

	#Get sentiment for each one
	print("--Sentimenting--")
	for ind in tweets.index:
		tweets.loc[ind,'sentiment_py']= get_sentiment(tweets.loc[ind,'token_text'])
		
	tweets.reset_index(inplace=True)

	#saving sentiment as a csv for cleaned tweets of a given ticker
	tweets.to_csv(os.path.join(PATH_RESULT,currentDT+'_sentiment.csv'), index=False)

	plot_result = tweets.groupby(['Date_Ext', 'sentiment_py']).size().unstack(fill_value=0)

	plot_result.reset_index(inplace=True)

	delta_sentiment = []

	#calculate percentage of each sentiment
	for idx in plot_result.index:
		plot_result.loc[idx,'pos_per'] = 100*plot_result.loc[idx, 'positive']/(plot_result.loc[idx, 'positive']+plot_result.loc[idx, 'negative']+plot_result.loc[idx, 'neutral'])
		plot_result.loc[idx,'neg_per'] = 100*plot_result.loc[idx, 'negative']/(plot_result.loc[idx, 'positive']+plot_result.loc[idx, 'negative']+plot_result.loc[idx, 'neutral'])
		plot_result.loc[idx,'neu_per'] = 100*plot_result.loc[idx, 'neutral']/(plot_result.loc[idx, 'positive']+plot_result.loc[idx, 'negative']+plot_result.loc[idx, 'neutral'])

	#calculate delta of sentiment day and day-1
	for idx in range(len(plot_result)):
		if idx == max(range(len(plot_result))):
			break
		else:
			positive_delta = plot_result.loc[idx+1, 'pos_per']-plot_result.loc[idx, 'pos_per']
			negative_delta = plot_result.loc[idx+1, 'neg_per']-plot_result.loc[idx, 'neg_per']
			avg_delta = (positive_delta+negative_delta)/2
			delta_sentiment.append(avg_delta)


	plot_original_result = tweets.groupby(['Date_Ext', 'Category']).size().unstack(fill_value=0)

	plot_original_result.reset_index(inplace=True)

	#calculate percentage of each sentiment
	for idx in plot_original_result.index:
		plot_original_result.loc[idx,'pos_ori_per'] = 100*plot_original_result.loc[idx, 'positive']/(plot_original_result.loc[idx, 'positive']+plot_original_result.loc[idx, 'negative']+plot_original_result.loc[idx, 'neutral'])
		plot_original_result.loc[idx,'neg_ori_per'] = 100*plot_original_result.loc[idx, 'negative']/(plot_original_result.loc[idx, 'positive']+plot_original_result.loc[idx, 'negative']+plot_original_result.loc[idx, 'neutral'])
		plot_original_result.loc[idx,'neu_ori_per'] = 100*plot_original_result.loc[idx, 'neutral']/(plot_original_result.loc[idx, 'positive']+plot_original_result.loc[idx, 'negative']+plot_original_result.loc[idx, 'neutral'])

	#calculate delta of sentiment day and day-1
	for idx in range(len(plot_original_result)):
		if idx == max(range(len(plot_original_result))):
			break
		else:
			positive_delta = plot_original_result.loc[idx+1, 'pos_ori_per']-plot_original_result.loc[idx, 'pos_ori_per']
			negative_delta = plot_original_result.loc[idx+1, 'neg_ori_per']-plot_original_result.loc[idx, 'neg_ori_per']
			avg_delta = (positive_delta+negative_delta)/2
			delta_sentiment.append(avg_delta)

	return delta_sentiment
	
if __name__ == "__main__":

	nlp = spacy.load('en_core_web_sm')

	tweets = pd.DataFrame()

	#find benchmark
	for filename in os.listdir(PATH_DATA):
		print("--Data Loading--")
		data = pd.read_csv(PATH_DATA+'/'+filename, low_memory=False, encoding='utf-8')
		tweets = tweets.append(data)
	benchmark(data)

	for filename in os.listdir(PATH_DATA):
		print("--Data Loading--")
		data = pd.read_csv(PATH_DATA+'/'+filename, low_memory=False, encoding='utf-8')
		pipeline(data)
		# print(data)
		

	


