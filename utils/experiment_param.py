import numpy as np
import pandas as pd


class ExperimentParam:

    def __init__(self, dirpath):

        # load metadata
        self.metadata = pd.read_csv(dirpath + 'EDF10d_info_2026_01_16.csv')

        # create new column, called merged id 
        self.metadata['Merged_ID'] = (
            self.metadata['Mouse ID'].str.strip("'") + '_' +
            self.metadata['FOV'].str.strip("'") + '_' +
            self.metadata['Session1'].str.strip("'").str.replace("-", "_") + '_' +
            self.metadata['Session2'].str.strip("'").str.replace("-", "_")
        )
        cols = ['Merged_ID'] + [
            c for c in self.metadata.columns if c != 'Merged_ID'
        ]
        self.metadata = self.metadata[cols]
        self.merged_ID = self.metadata['Merged_ID']

        # sort by relearn speed
        self.colname_relearn_speed = 'Relative trials to reach\n75% performance'
        self.relearn_time = self.metadata[self.colname_relearn_speed]
        self.sorted_indices_by_relearn_speed = np.argsort(self.relearn_time)

        # trial info
        self.nfile = len(self.metadata)
        self.acts = ['P1', 'A1', 'P2', 'A2']
        self.lickright = 0
        self.lickleft  = 1

        # time info
        self.tsample   = 1.57
        self.tdelay    = 2.87
        self.tresponse = 4.17
        self.dt = 1/6
        self.ntimestep = 47

        self.tvec = self.dt * np.arange(self.ntimestep)
        self.tix_sample = np.where(self.tvec > self.tsample)[0][0]
        self.tix_delay = np.where(self.tvec > self.tdelay)[0][0]
        self.tix_response = np.where(self.tvec > self.tresponse)[0][0]
        
        self.tix_start = self.tix_sample
        self.tix_end   = self.tix_response + 8
        self.t = self.tvec[self.tix_start:self.tix_end]
        self.tgo = (self.t - self.tvec[self.tix_response])

        self.keys1 = ['P1+A1-', 'P1+A1+', 'P1-A1+']
        self.keys2 = ['P2+A2-', 'P2+A2+', 'P2-A2+']
        self.keys3 = ['P1','A1','P2','A2']
        self.keys_within_context = ['P1-A1','P2-A2']
        self.keys_across_context = ['P2-P1','A2-A1']




