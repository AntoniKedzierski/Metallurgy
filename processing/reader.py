import pandas as pd
import os
import json

class File:
    def __init__(self, path):
        self.path = path
        self.data = pd.DataFrame()

    def read(self):
        pass

    def print(self):
        print(f'Plik o ścieżce {self.path}')


class ExcelFile(File):
    def __init__(self, path, load=False, **kwargs):
        super().__init__(path)
        if load:
            usecols = None
            first_row = 1
            if 'usecols' in kwargs:
                usecols = kwargs['usecols']
            if 'first_row' in kwargs:
                first_row = kwargs['first_row']
            self.read(usecols=usecols, first_row=first_row)

    def read(self, usecols, first_row):
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f'Nie istnieje plik o ścieżce {self.path}')
        self.data = pd.read_excel(self.path, usecols=usecols, skiprows=(first_row - 1))

    def print(self, columns='all', wide_view=False, nrow=20, ncol=50):
        display_columns = list(self.data.columns)
        if isinstance(columns, list):
            display_columns = columns
        elif columns != 'all':
            raise ValueError('Błędny wartość parametru columns. Musi być to lista lub "all".')
        if wide_view:
            with pd.option_context('display.min_rows', nrow, 'display.max_columns', ncol, 'display.width', 1000):
                print(self.data.loc[:, display_columns])
        else:
            print(self.data.loc[:, display_columns])
