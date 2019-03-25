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


currentDT = f"{datetime.now():%Y-%m-%d-%H-%M}"

PATH_DATA = 'data'
PATH_RESULT = 'result'

nltk.download('wordnet')

#Get the stop words
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

nlp = spacy.load('en_core_web_sm')
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

	if type(tweet) != list:
		result = 'unknown'
		# print(result)
		return result
	analysis = TextBlob(' '.join(tweet))
	# set sentiment 
	if analysis.sentiment.polarity > 0:
		result = 'positive'
	elif analysis.sentiment.polarity == 0:
		result = 'neutral'
	else: 
		result = 'negative'

	# print(result)
	return result


def date_plot(date, ypos, yneg, yneu, yunk, title):
	# create the plot space upon which to plot the data
	fig, ax= plt.subplots(1, 1, figsize=(16, 9), dpi=100)

	# add the x-axis and the y-axis to the plot
	# ax.plot(date, ypos, 'positive', yneg, 'negative', yneu, 'neutral', yunk, 'unknown', color = 'red')
	ax.plot(date, ypos, label='positive', color = 'red')
	ax.plot(date, yneg, label='negative', color = 'blue')
	ax.plot(date, yneu, label='neutral', color = 'green')
	ax.plot(date, yunk, label='unknown', color = 'black')

	# rotate tick labels
	plt.setp(ax.get_xticklabels(), rotation=25)

	# set title and labels for axes
	ax.set(xlabel="Date",
       ylabel="Sentiment",
       title="Playstation Now Sentiment, from "+min(date)+" to "+max(date));

	# Place a legend to the right of this smaller subplot.
	plt.legend(bbox_to_anchor=(1, 1), loc=2)

	plt.savefig('plot/'+currentDT+'_'+title+'_result_plot.png')

    

if __name__ == "__main__":

	tweets = pd.DataFrame()

	for filename in os.listdir(PATH_DATA):
		data = pd.read_csv(PATH_DATA+'/'+filename, low_memory=False)
		# print(data)
		tweets = tweets.append(data)

	tweets = tweets.loc[:,['Date','Full Text','Domain']]

	# tweets['token_text'] = [tokenize(item['Full Text']) for item in tweets]
	for i in range(len(tweets)):
	    #Separate each line to text_data list
	    tweets.loc[i,'Date'] = str(tweets.loc[i,'Date'])[0:23]
	    date , time = datetime_extraction(str(tweets.loc[i,'Date'])[0:23])
	    tweets.loc[i,'Date_Ext'] = date
	    tweets.loc[i,'Time_Ext'] = time


	# tweets['Added'] = tweets.loc[:,['Added']][0:23]

	tweets.drop_duplicates(inplace=True)

	print(tweets[0:6])

	#Tokenizing
	print("--Tokenizing--")
	result_token = []
	# tweets['token_text'] = [tokenize(item['Full Text']) for item in tweets]
	for ind in tweets.index:
	    #Separate each line to token list
	    text = tweets.loc[ind,'Full Text']
	    if pd.isnull(text):
	    	result_token.append([])
	    elif (str(text).strip() != ""):
	    	# print(text)
	    	token = tokenize(tweets.loc[ind,'Full Text'])
	    	# print(token)
	    	result_token.append(token)
	    else:
	    	result_token.append([])
	    # tweets.at[i,'token_text'] = token
	# print(text_data)

	tweets['token_text'] = result_token

	tweets.set_index(['Date'],inplace=True)

	#Get sentiment for each one
	print("--Sentimenting--")
	for ind in tweets.index:
		# tweets.loc[i,'tweetText'] = clean_tweet(tweets.loc[i,'tweetText'])
		tweets.loc[ind,'sentiment_py']= get_sentiment(tweets.loc[ind,'token_text'])
    
	#saving sentiment as a csv for cleaned tweets of a given ticker
	tweets.to_csv(os.path.join(PATH_RESULT,currentDT+'_sentiment.csv'), index=False)
    
	#returning a list of [positive, neutral, negative]
	# print(tweets['sentiment_py'].value_counts())

	print(tweets[0:6])

	plot_result = tweets.groupby(['Date_Ext', 'sentiment_py']).size().unstack(fill_value=0)

	print(plot_result[0:6])

	plot_result.reset_index(inplace=True)

	#saving sentiment as a csv for cleaned tweets of a given ticker
	plot_result.to_csv(os.path.join(PATH_RESULT,currentDT+'_sentiment_count.csv'), index=False)

	#Plot
	date_plot(plot_result['Date_Ext'], plot_result['positive'], plot_result['negative'], plot_result['neutral'], plot_result['unknown'], 'overall')



	# for i in range(len(tweets)):
	# 	tweets.loc[i,'tweetText'] = clean_tweet(tweets.loc[i,'tweetText'])

	# print(tweets[0:6])


