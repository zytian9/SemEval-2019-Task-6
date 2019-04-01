from __future__ import division 
import csv 
import string 
from nltk.tokenize import TweetTokenizer
import json 
from nltk.tokenize import MWETokenizer
import re
import emoji
import spacy

def average_len(line):
	split_line = line.split(' ')
	avg = '{0:.2f}'.format(float(sum(len(word) for word in split_line)/len(split_line)))
	return avg

def find_punc(line):
	a_punct = count(line, string.punctuation)
	return a_punct

def con(MWE):
	count_conn = 0
	for word in MWE:
		if word.lower() in disc_conn:
			count_conn +=1	
	return count_conn

def mod(MWE):
	count_mod = 0
	for word in MWE:
		if word.lower() in modals:
			count_mod +=1	
	return count_mod

def count_one(line):
	line = re.sub(r'[^\w\s]','',line)
	split_line = line.split(' ')
	count = 0
	listW = []
	tweet = [re.sub(r"^[0-9_]*$", '', x) for x in split_line]
	for word in tweet:
		if len(word) == 1:
			count+=1
			listW.append(word)
	return count

def find_url(line):
	url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', line)
	count_url = 0
	if url:
		count_url+=1
	return count_url

def ent_type(line):
	doc=nlp(line)
	count_grp = 0
	count_person=0
	for ent in doc.ents:
		if ent.label_ == "GPE" or "ORG" or "NORP":
			count_grp +=1
		elif ent.label_ == "PERSON":
			count_person +=1
	return count_grp, count_person, len(doc.ents)

def emoji_sentiment (line):
    count_emoji = 0
    emoj_neg = 0
    emoj_pos = 0
    emoj_all = 0
    for word in line:
        for char in word:
            if char in emoji.UNICODE_EMOJI:
                count_emoji +=1
                if char in sent_dic.keys():
                    emoj_neg+=float(sent_dic[char][0])
                    emoj_pos+=float(sent_dic[char][1])
                    emoj_all+=float(sent_dic[char][2])
    return count_emoji, emoj_neg, emoj_pos, emoj_all

def emoticon_sentiment (line):
    emoticon = re.findall(':\)|:-\)|:\(|:-\(|;\);-\)|:-O|8-|:P|:D|:\||:S|:\$|:@|8o\||\+o\(|\(H\)|\(C\)|\(\?\)', line)
    pos_emo = [':)' , '(:' , ';)' , ':-)' , '(-:' ,':D ',':-D' , ':P' , ':-P']
    neg_emo =  [':(' , '):', ';(', ':-(', ')-:' , 'D:' , 'D-:' , ':’( ', ':’-( ', ')’: ', ')-’:']
    num_emotic = len(emoticon)
    emot_pos = 0
    emot_neg = 0
    for i in emoticon:
        if i in pos_emo:
            emot_pos +=1
        elif i in neg_emo:
            emot_neg +=1
    return num_emotic, emot_pos,emot_neg


def main():
	nlp = spacy.load('en_core_web_sm')
	t_conn = MWETokenizer(
		[('although'), ('in', 'turn'), ('afterward'), ('consequently'), ('additionally'), ('alternatively'),
		 ('whereas'), ('on', 'the', 'contrary'), ('if', 'and', 'when'), ('lest'), ('and', 'on', 'the', 'one', 'hand'),
		 ('on', 'the', 'other', 'hand')])
	t_modals = MWETokenizer(
		[('be'), ('is'), ('am'), ('are'), ('were'), ('was'), ('been'), ('being'), ('he\'s'), ('I\'m'), ('it\'s'),
		 ('she\'s'), ('they\'re'), ('we\'re'), ('you\'re'), ('that\'s'), ('aren\'t'), ('isn\'t'), ('wasn\'t'),
		 ('weren\'t'), ('am'), ('is'), ('are'), ('was'), ('were'), ('will'), ('might\'ve'), ('may\'ve'), ('may'),
		 ('might'), ('mightn\'t'), ('mayn\'t'), ('cannot'), ('can\'t'), ('couldn\'t'), ('can'), ('could'),
		 ('could\'ve'), ('should\'ve'), ('shalt'), ('shan\'t'), ('shall'), ('should'), ('shouldn\'t'), ('able'),
		 ('unable'), ('ain\'t'), ('you\'d'), ('he\'d'), ('I\'d'), ('it\'d'), ('she\'d'), ('they\'d'), ('we\'d'),
		 ('that\'d'), ('this\'d'), ('i\'ve'), ('you\'ve'), ('she\'s'), ('he\'s'), ('we\'ve'), ('they\'ve'), ('it\'s'),
		 ('that\'s'), ('this\'s'), ('will'), ('would'), ('wouldn\'t'), ('won\'t'), ('i\'ll'), ('you\'ll'), ('she\'ll'),
		 ('he\'ll'), ('that\'ll'), ('this\'ll'), ('daren\'t'), ('dare'), ('has'), ('had'), ('have'), ('hasn\'t'),
		 ('gotta'), ('got', 'to'), ('must\'ve'), ('must'), ('mustn\'t'), ('mustn\'t \'ve'), ('need'), ('needn\'t'),
		 ('need\'ve'), ('need', 'have'), ('needn\'t', 'have'), ('needn\'t\'ve'), ('need', 'not', 'have'),
		 ('oughtn\'t', 'to'), ('ought'), ('oughta'), ('do'), ('does'), ('doesn\'t'), ('don\'t'), ('going', 'to'),
		 ('gonna')])

	count_nonAlpha = 0
	count_insult = 0

	# get sentiment score from Emoji Sentiment Rank, namely, negative, positive, overall
	# http://kt.ijs.si/data/Emoji_sentiment_ranking
	sent_file = open('./emoji_sentiment_rank', encoding='utf-8')
	sent_corpus = [line.split('\t') for line in sent_file]
	sent_dic= {}
	for i in sent_corpus:
		sent_dic[i[1]] = (i[-6],i[-4],i[-3])

	count_capt = 0

	tknzr = TweetTokenizer()
	# path = './trial-data/trial-taskc-emoji.tsv'
	# ./trial-data/offenseval-trial-taskc-tweet.tsv
	path = './training-data/training-taskc-emoji.tsv'
	csv_writer = open(path,'w',encoding = 'utf-8')

	for line in open('./training-data/offenseval-training-taskc-tweet.tsv', encoding = 'utf-8'):
		tweet = line.strip()
		tokenized_tweet = tknzr.tokenize(tweet)
		avg_len = average_len(tweet)
		punc = find_punc(tweet)
		one_letter = count_one(tweet)
		url =find_url(tweet)
		count_emoji, emoj_neg, emoj_pos, emoj_all = emoji_sentiment(tweet)
		num_emotic, emot_neg, emot_pos= emoticon_sentiment(tweet)
		conn_MWE = t_conn.tokenize(tweet.split())
		counted_c = con(conn_MWE)
		mod_MWE = t_modals.tokenize(tweet.split())
		counted_m = mod(mod_MWE)
		# entity numbers
		count_grp, count_person,count_ent = ent_type(tweet)
		for word in tweet.split(' '):
			# tokens with non-alpha characters in the middle
			non_alpha = re.match('[a-zA-Z]+[^a-zA-Z\d\s:]+[a-zA-Z]+', word)
			capt_let = re.match('[A-Z]+',word)
			if capt_let:
				count_capt +=1
			if non_alpha:
				count_nonAlpha+=1
			if word.lower() in insults:
				count_insult +=1
		csv_writer.write(str(len(tokenized_tweet))+'\t'+str(avg_len)+'\t'+str(punc)+'\t'+str(one_letter)+'\t'+str(count_capt)+'\t'+str(url)+'\t'+str(count_nonAlpha)+'\t'+str(counted_c)+'\t'+str(counted_m)+'\t'+str(count_emoji)+'\t'+str(emoj_neg)+'\t'+str(emoj_pos)+'\t'+str(emoj_all)+'\t'+str(num_emotic)+'\t'+str(emot_neg)+'\t'+str(emot_pos))
		csv_writer.write('\n')
		count_nonAlpha = 0
		count_insult = 0
		count_capt = 0


if __name__ == '__main__':
	main()
