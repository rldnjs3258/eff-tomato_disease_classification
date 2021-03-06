""" Weighted Voting 앙상블 코드

TODO:

NOTES:

REFERENCE:

UPDATED:
"""

import pandas as pd
import numpy as np
from scipy.special import softmax

class WeightedVotingEnsemble():
    def __init__(self, df, key, label):
        self.df = df
        self.key = key
        self.label = label
    
    # weight 설정
    def __set_weight__(self, pred, weight):
        for i in range(weight):
            self.df = pd.merge(self.df, pred, on=self.key)
    
    # ensemble
    def __ensemble__(self):
        vote = self.df.mode(axis='columns').to_numpy()
        ensemble_result = vote[:, 0]
        return ensemble_result
    
    # submission 파일 만들기
    def __submission__(self, ensemble_result):
        submission = pd.DataFrame({'file_name': self.df[self.key],
                                  'answer': ensemble_result.flatten()})
        answer = submission['answer'].apply(np.int64)
        submission = pd.DataFrame({'file_name': self.df[self.key],
                                  'answer': answer})
        print('Weighted voting ensemble done!')
        print(submission.head())
        return submission

    def __to_csv__(self, path, submission):
        submission.to_csv(path, index=False)


if __name__ == '__main__':
    df = pd.read_csv('results/csv/[0.9809]Efficientb0-layer(1280-500-250-10)-ES(20)-IS(224)_Aug(NoColor).csv')
    pred1 = pd.read_csv('results/csv/[0.9953]Efficientnetb6-layer(1280-500-250-10)-ES(50)-IS(528)_Aug(NoColor).csv')
    pred2 = pd.read_csv('results/csv/[0.9892]Efficientb0-layer(1280-500-250-10)-ES(50)-IS(224)_Aug(NoColor).csv')
    path = 'results/csv/ensemble_result2.csv' # 최종 ensemble 결과 저장 위치
    
    # set parmas
    weight = {'pred1' : 3,
              'pred2' : 2} # weight
    key = 'file_name'
    label = 'answer'
    
    # class : WeightVotingEnsemble
    weighted_voting_ensemble = WeightedVotingEnsemble(df, key, label)
    weighted_voting_ensemble.__set_weight__(pred1, weight['pred1'])
    weighted_voting_ensemble.__set_weight__(pred2, weight['pred2'])
    ensemble_result = weighted_voting_ensemble.__ensemble__()
    submission = weighted_voting_ensemble.__submission__(ensemble_result)
    weighted_voting_ensemble.__to_csv__(path, submission)