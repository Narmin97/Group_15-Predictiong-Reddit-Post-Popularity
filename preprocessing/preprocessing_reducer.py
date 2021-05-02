#!/usr/bin/env python3

"""
Preprocessing Reducer

"""

import sys

for line in sys.stdin:
    line = line.strip().split("\t")
    print ("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s" % (line[0], line[1], line[2], 
                                                    line[3], line[4], line[5], 
                                                    line[6], line[7], line[8], 
                                                    line[9], line[10], line[11])) 