import pandas as pd
import numpy as np
import os.path
SLICE_SIZE=1

class Reader:

    total = 0
    current = 0
    df = pd.DataFrame()

    def __init__(self):
        self.read_file()

    def read_file(self):
        file = 'sample.csv'
        if (os.path.exists("dataset.csv")):
            file = 'dataset.csv'
        self.df = pd.read_csv(file)
        self.total = len(self.df.index)
        print(f'total lines: {self.total}  in file: {file}')

    # def get_next_row(self):
    #     print('get_next_row called: df is',self.df)
    #     row = self.df.iloc[self.current]
    #     print("current row:", row)
    #     self.current += 1
    #     return [float(row.latitude),float(row.longitude),float(row.heading),float(row.timestamp)]

    def get_fixed_slice(self):
        print(f'total: {self.total}  current: {self.current}' )
        if ((self.total - self.current) / SLICE_SIZE >= 1):
            slice = self.df.iloc[self.current : self.current + SLICE_SIZE]
            print(f'slice: \n {slice}')
            print(f'values: \n {slice.values.tolist()}')
            self.current += SLICE_SIZE
            return slice.values.tolist()

    def reset(self):
        print('read csv again')
        self.current=0
        self.read_file()

    # def init_count(self):
    #     return self.total_rows

# if __name__ == '__main__':
#     reader = Reader()
#     reader.get_fixed_slice()
#     reader.get_fixed_slice()