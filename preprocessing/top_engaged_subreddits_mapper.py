#!/usr/bin/env python3

"""
Engagement Score Calculation Mapper

"""

import sys
import csv

csv.field_size_limit(sys.maxsize)

def PreProcessingMapper(argv):

    # Gets the first line and does nothing with it.
    my_iterator = iter(sys.stdin.readline, "")
    next(my_iterator)
    # For each row in the dataset
    for row in csv.reader(iter(sys.stdin.readline, "")):
        if len(row) == 95:
            # Replaces NaN with 0
            for i in range(0, 95):
                 if isinstance(row[i], int):
                     continue
                 elif len(row[i]) > 0:
                     continue
                 else:
                     row[i]=0
            # Collects the values to dictionary
            author = row[2]
            link = row[15]
            num_comments = row[32]
            subreddit = row[47]

            if author != '[deleted]':
                # Printing the result of Preprocessing
                print("%s\t%s\t%s\t%s" % (author, link, num_comments, subreddit))
            else:
                continue

            # Printing the result of Preprocessing
           # print("%s\t%s\t%s\t%s\t%s\t%s" % (author, utc, link, num_comments, score, subreddit))
        else:
            continue

if __name__ == "__main__":
    PreProcessingMapper(sys.argv)
