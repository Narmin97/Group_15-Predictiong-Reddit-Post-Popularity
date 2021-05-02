#!/usr/bin/env python3

"""
Preprocessing Mapper

"""

import sys
import csv
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def PreProcessingMapper(argv):
    sid = SentimentIntensityAnalyzer()
    # Gets the first line and does nothing with it.
    my_iterator = iter(sys.stdin.readline, "")
    next(my_iterator)
    # For each row in the dataset
    for row in csv.reader(iter(sys.stdin.readline, "")):
        if len(row) ==22:
            # Replaces NaN with 0
            for i in range(0, 22):
                 if isinstance(row[i], int):
                     continue
                 elif len(row[i]) > 0:
                     continue
                 else:
                     row[i]=0

            timestamp = row[0]
            utc = pd.to_datetime(timestamp, unit='s')
            hour = utc.hour
            dayofweek = utc.isoweekday()
            day = utc.day
            month = utc.month
            year = utc.year

            # Applies sentiment Analyses
            metrics = {}
            def remove_int(text):
                return ''.join([str(i) for i in text])
            if isinstance(row[17], int):
                sentiment_class = 10
            else:

                ss = sid.polarity_scores(remove_int(row[17]))

                for k in sorted(ss):
                    metrics[k] = ss[k]

                # Divides the Body into Sentiment Classes : {1: 'HP', 2: 'MP', 3: 'N', 4: 'MN', 5: 'HN'}
                if(metrics['compound'] > 0.6):
                    sentiment_class = 1
                elif(metrics['compound'] > 0.25):
                    sentiment_class = 2
                elif(metrics['compound'] > -0.25):
                    sentiment_class = 3
                elif(metrics['compound'] > -0.6):
                    sentiment_class = 4
                else:
                    sentiment_class = 5
            # Printing the result of Preprocessing
            print("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (row[9], row[11], row[13], 
                                                                      sentiment_class, row[19], 
                                                                      row[20], hour, dayofweek, 
                                                                      day, month, year, row[15]))
        else:
            continue
