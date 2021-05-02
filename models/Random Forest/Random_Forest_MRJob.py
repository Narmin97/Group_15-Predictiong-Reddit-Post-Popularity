"""
Random Forest MRJob.

"""

from mrjob.job import MRJob
import random
import csv
import numpy as np
import sys
from datetime import datetime

sys.setrecursionlimit(10**9)

class DecisionTree:


    def __init__(self, training_data, field_types):

        self.training_data = np.array(training_data)
        self.field_types = field_types

    def mean_squared_error(self, records):
        if records.shape[0] == 0:
            return 0
        targets = records[:, -1].astype('float')
        value = np.mean(targets)
        mse = np.mean(np.square(targets - value))
        return mse
    
    def find_best_attr(self, records):
        result = {'attr': None, 'split_cond': None, 'splits': None}
        min_mse = -1

        for i in np.arange(0, records.shape[1] - 1):

            split_cond = None
            splits = {}

            if self.field_types[i] == 'N':

                left_split = list()
                right_split = list()
                split_cond = np.mean(records[:, i].astype('float'))

                for record in records:
                    if record[i] < split_cond:
                        left_split.append(record)
                    else:
                        right_split.append(record)
                splits['left'] = np.array(left_split)
                splits['right'] = np.array(right_split)
            else:
                
                split_cond = list(set(records[:, i]))
                splits = {cond:list() for cond in split_cond}
                for record in records:
                    splits[record[i]].append(record)
                splits = {k: np.array(v) for k,v in splits.items()}

            error = 0
            for cond in splits:
                split = splits[cond]
                error += (split.shape[0]/records.shape[0])*self.mean_squared_error(split)

            if min_mse == -1 or error < min_mse:
                result['attr'] = i
                result['split_cond'] = split_cond
                result['splits'] = splits
                min_mse = error

        return result
    
    def split(self, node):

        splits = node['splits']
        min_record = 1
        for i in splits:
            split = splits[i]
            if split.shape[0] <= min_record:
                node[i] = np.mean(split[:, -1].astype('float'))
            else:
                node[i] = self.find_best_attr(split)
                self.split(node[i])
                
    def build_model(self):
        root_node = self.find_best_attr(self.training_data)
        self.split(root_node)
        return root_node
    
    def apply_model(self, node, record):
        if self.field_types[node['attr']] == 'N':
            if record[node['attr']] < node['split_cond']:
                if isinstance(node['left'], dict):
                    return self.apply_model(node['left'], record)
                else:
                    return node['left']
            else:
                if isinstance(node['right'], dict):
                    return self.apply_model(node['right'], record)
                else:
                    return node['right']
        else:
            cat = record[node['attr']]
            if cat not in node['split_cond'] and len(node['split_cond']) > 0:

                cat = node['split_cond'][0]

            if isinstance(node[cat], dict):
                return self.apply_model(node[cat], record)
            else:
                return node[cat]
    
    def predict(self, model, test_data):
        predictions = []
        for record in test_data:
            pred_val = self.apply_model(model, record)
            predictions.append([pred_val])
        return predictions
    
class MRPredictDelay(MRJob):

    field_types = ['C', 'C', 'C', 'C', 'C', 'C', 'C', 'N']

    def configure_options(self):
        super(MRPredictDelay, self).configure_options()
        self.add_passthrough_option('--maxSplitNumber', type='int', default=10)
        self.add_passthrough_option('--sampleNumber', type='int', default=1000)
        self.add_passthrough_option('--sampleSize', type='int', default=1000000)
        self.add_file_option('--testData')
        
    def steps(self):
        return [
            self.mr(mapper = self.step_1_mapper,
                    reducer = self.step_1_reducer),
            self.mr(mapper_init = self.step_2_test_mapper,
                    mapper = self.step_2_mapper,
                    combiner = self.step_2_combiner,
                    reducer = self.step_2_reducer)]
    
    def step_1_mapper(self,_,line):
        key = random.randint(0, self.options.maxSplitNumber)
        line = line.replace('\t', '').split(',')
        yield key, line
        
    def step_1_reducer(self, key, values):
        values_list = np.array(list(values))
        l = values_list.shape[0]
        if l < self.options.sampleSize:
            yield (key, list(values))
        else:
            for i in range(0, self.options.sampleNumber):
                idx = np.random.choice(l, size=self.options.sampleSize, replace=False)
                yield "{}_{}".format(key, i), values_list[idx, :].tolist()
                
    def step_2_test_mapper(self):
        self.test_set = []
        with open(self.options.testData) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                 self.test_set.append(row)
    
    def step_2_mapper(self, key, trainingData):
        DT = DecisionTree(training_data = trainingData, field_types = self.field_types)
        model = DT.build_model()
        predictions = DT.predict(model, self.test_set)
        yield 'predictions',(1,predictions)
        
    def step_2_combiner(self, key, predictions):
        predictions = list(predictions)
        combined_prediction = []
        predictions_number = len(predictions)
        for i in range(0, predictions_number):
            if i == 0:
                combined_prediction = predictions[i][1]
            else:
                combined_prediction = np.add(combined_prediction, predictions[i][1])
        yield key, (predictions_number, combined_prediction.tolist())
        
    def step_2_reducer(self, key, predictions):
        predictions = list(predictions)
        final_prediction = []
        predictions_number = 0
        for i in range(0, len(predictions)):
            predictions_number += predictions[i][0]
            if i == 0:
                final_prediction = predictions[i][1]
            else:
                final_prediction = np.add(final_prediction, predictions[i][1])

        final_prediction = np.divide(final_prediction,predictions_number)
        yield key, final_prediction.tolist()
        
if __name__ == '__main__':
    start_time=datetime.now()
    MRPredictDelay.run()
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    sys.stderr.write(str(elapsed_time))
        