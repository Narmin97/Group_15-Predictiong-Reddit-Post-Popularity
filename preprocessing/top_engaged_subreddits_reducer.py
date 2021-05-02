#!/usr/bin/env python3

"""
Engagement Score Calculation Reducer

"""

import sys

link_comments = {}
subreddit_link = {}
subreddit_author = {}
new = {}

#Partitoner
for line in sys.stdin:
    line = line.strip()
    line = line.split('\t')

    author = line[0]
    link = line[1]
    num_comments = line[2]
    subreddit = line[3]

    if link in link_comments:
        link_comments[link].append(int(num_comments))
    else:
        link_comments[link] = []
        link_comments[link].append(int(num_comments))

    if subreddit in subreddit_author:
        subreddit_author[subreddit].append(str(author))
    else:
        subreddit_author[subreddit] = []
        subreddit_author[subreddit].append(str(author))

    if subreddit in subreddit_link:
        subreddit_link[subreddit].append(str(link))
    else:
        subreddit_link[subreddit] = []
        subreddit_link[subreddit].append(str(link))
        
sum_comments = {}
for link in link_comments.keys():
    sum_comments[link] = []
    sum_comments[link].append(sum(link_comments[link]))


for key in subreddit_link.keys():
    for value in subreddit_link[key]:
        if key in new:
            new[key].append(int(str(sum_comments[value])[1:-1]))
        else:
            new[key] = []
            new[key].append(int(str(sum_comments[value])[1:-1]))

for subreddit in subreddit_author.keys():
    num_author = len(subreddit_author[subreddit])
    num_links = sum(new[subreddit])

    print ('%s,%s,%s,' % (subreddit, num_author, num_links))