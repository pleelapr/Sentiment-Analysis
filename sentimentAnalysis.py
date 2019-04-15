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

TO_INCLUDE = True
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
	elif analysis.sentiment.polarity < 0:
		result = 'negative'
	else: 
		result = 'neutral'

	# print(result)
	return result

def plot(date, lbl1, lbl2, title):
	# create the plot space upon which to plot the data
	fig, ax= plt.subplots(1, 1, figsize=(16, 9), dpi=100)

	# add the x-axis and the y-axis to the plot
	# ax.plot(date, ypos, 'positive', yneg, 'negative', yneu, 'neutral', yunk, 'unknown', color = 'red')
	ax.plot(date, lbl1, label='retweet', color = 'red')
	ax.plot(date, lbl2, label='not-retweet', color = 'blue')

	# rotate tick labels
	plt.setp(ax.get_xticklabels(), rotation=25)

	# plt.text(0.02, 0.5, PLOT_NOTE, fontsize=14, transform=plt.gcf().transFigure)

	# Now let's add your additional information
	ax.annotate(PLOT_NOTE, xy=(0.5, 0), xytext=(0, 0), xycoords=('axes fraction', 'figure fraction'), textcoords='offset points', size=14, ha='center', va='bottom')

	# set title and labels for axes
	ax.set(xlabel="Date",
       ylabel="Count",
       title="Playstation Now Retweet Count, from "+min(date)+" to "+max(date));

	# Place a legend to the right of this smaller subplot.
	plt.legend(bbox_to_anchor=(1, 1), loc=2)

	plt.savefig('plot/'+currentDT+'_'+title+'_retweet_plot.png')

def sentiment_plot(date, ypos, yneg, yneu, title):
	# create the plot space upon which to plot the data
	fig, ax= plt.subplots(1, 1, figsize=(16, 9), dpi=100)

	# add the x-axis and the y-axis to the plot
	# ax.plot(date, ypos, 'positive', yneg, 'negative', yneu, 'neutral', yunk, 'unknown', color = 'red')
	ax.plot(date, ypos, label='positive', color = 'red')
	ax.plot(date, yneg, label='negative', color = 'blue')
	ax.plot(date, yneu, label='neutral', color = 'green')
	# ax.plot(date, yunk, label='unknown', color = 'black')

	# rotate tick labels
	plt.setp(ax.get_xticklabels(), rotation=25)

	# plt.text(0.02, 0.5, PLOT_NOTE, fontsize=14, transform=plt.gcf().transFigure)
	ax.annotate(PLOT_NOTE, xy=(0.5, 0), xytext=(0, 0), xycoords=('axes fraction', 'figure fraction'), textcoords='offset points', size=14, ha='center', va='bottom')

	# set title and labels for axes
	ax.set(xlabel="Date",
       ylabel="Sentiment",
       title="Playstation Now Sentiment, from "+min(date)+" to "+max(date));

	# Place a legend to the right of this smaller subplot.
	plt.legend(bbox_to_anchor=(1, 1), loc=2)

	plt.savefig('plot/'+currentDT+'_'+title+'_result_plot.png')


if __name__ == "__main__":

	tweets = pd.DataFrame()
	print("--Data Loading--")
	for filename in os.listdir(PATH_DATA):
		data = pd.read_csv(PATH_DATA+'/'+filename, low_memory=False, encoding='cp1252')
		# print(data)
		tweets = tweets.append(data)

	tweets = tweets.loc[:,['Date','Full Text','Domain','Sentiment','Twitter Retweet of']]

	print(tweets[0:6])

	tweets.loc[:,'Date_Ext'] = tweets.loc[:,'Date'].str.slice(0,11)
	tweets.loc[:,'Time_Ext'] = tweets.loc[:,'Date'].str[12,]

	print(tweets.loc[0:6,'Date_Ext'])
	# tweets['token_text'] = [tokenize(item['Full Text']) for item in tweets]
	# 12:

	tweets.drop_duplicates(inplace=True)

	print(tweets['Domain'].value_counts())

	domain_count = tweets.groupby(['Date_Ext', 'Domain']).size().unstack(fill_value=0)

	domain_count.to_csv(os.path.join(PATH_RESULT,currentDT+'_domain_count.csv'), index=True)
	#Plot
	# plot(domain_count['Date_Ext'], domain_count['positive'], domain_count['negative'], plot_result['neutral'], plot_result['unknown'], 'overall')
	
	#Get Only Topics
	if TO_INCLUDE:
		for ind in tweets.index:
			if any(word in str(tweets.loc[ind,'Full Text']).lower() for word in TOPIC):
				continue
			else:
				tweets.drop(ind, inplace=True)

	if TO_EXCLUDE:
		for ind in tweets.index:
			if any(word in str(tweets.loc[ind,'Full Text']).lower() for word in EXCLUDE_TOPIC):
				tweets.drop(ind, inplace=True)
			else:
				continue
		

	for ind in tweets.index:
	    #Separate each line to text_data list
		# print(len(str(tweets.loc[ind,'Twitter Retweet of'])))
		if 5 > len(str(tweets.loc[ind,'Twitter Retweet of'])) or "RT" == str(tweets.loc[ind,'Full Text']):
			tweets.loc[ind,'retweet'] = 'retweet'
		else:
			tweets.loc[ind,'retweet'] = 'not_retweet'

	retweet_count = tweets.groupby(['Date_Ext', 'retweet']).size().unstack(fill_value=0)
	
	print(retweet_count[0:6])

	retweet_count.reset_index(inplace=True)
	# Plot
	plot(retweet_count['Date_Ext'], retweet_count['retweet'], retweet_count['not_retweet'], 'retweet')

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
	print(type(result_token))

	#Get only specific topic
	

	tweets['token_text'] = result_token

	tweets.set_index(['Date'],inplace=True)

	#Get sentiment for each one
	print("--Sentimenting--")
	for ind in tweets.index:
		# tweets.loc[i,'tweetText'] = clean_tweet(tweets.loc[i,'tweetText'])
		tweets.loc[ind,'sentiment_py']= get_sentiment(tweets.loc[ind,'token_text'])
		print(tweets.loc[ind,'sentiment_py'], tweets.loc[ind,'Sentiment'])
		#compare sentiment with brandwatch result
		if type(tweets.loc[ind,'sentiment_py']) != str:
			tweets.loc[ind,'same_result_bw'] = 0
		elif tweets.loc[ind,'sentiment_py'].strip() == tweets.loc[ind,'Sentiment'].strip():
			tweets.loc[ind,'same_result_bw'] = 1
		else:
			tweets.loc[ind,'same_result_bw'] = 0
			
	
    
	#returning a list of [positive, neutral, negative]
	# print(tweets['sentiment_py'].value_counts())

	#compare sentiment with brandwatch
	# print(str(sum(tweets.loc[ind,'same_result_bw'])))

	#saving sentiment as a csv for cleaned tweets of a given ticker
	tweets.to_csv(os.path.join(PATH_RESULT,currentDT+'_sentiment.csv'), index=False)

	print(tweets[0:6])

	plot_result = tweets.groupby(['Date_Ext', 'sentiment_py']).size().unstack(fill_value=0)

	print(plot_result[0:6])

	plot_result.reset_index(inplace=True)

	#saving sentiment as a csv for cleaned tweets of a given ticker
	plot_result.to_csv(os.path.join(PATH_RESULT,currentDT+'_sentiment_count.csv'), index=False)

	#Plot
	sentiment_plot(plot_result['Date_Ext'], plot_result['positive'], plot_result['negative'], plot_result['neutral'], 'overall')



	# for i in range(len(tweets)):
	# 	tweets.loc[i,'tweetText'] = clean_tweet(tweets.loc[i,'tweetText'])

	# print(tweets[0:6])


