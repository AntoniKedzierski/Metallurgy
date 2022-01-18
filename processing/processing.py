import json

import numpy as np
import pandas as pd

from processing.reader import File


class DataObject: # --> do wydzielenia danych treningowych i walidacyjnych
    def __init__(self, file: File):
        self.data = file.data

    def refactor_columns_with_file(self, config_file):
        rename_dict = json.load(open(config_file, 'r', encoding='utf-8'))
        self.data = self.data.loc[:, list(rename_dict.keys())].rename(columns=rename_dict)

    def print(self, columns='all', rows='all', wide_view=False, nrow=20, ncol=50):
        display_columns = list(self.data.columns)
        display_rows = list(range(self.data.shape[0]))
        if isinstance(columns, list):
            display_columns = columns
            if isinstance(rows, list):
                display_rows = rows
            elif rows != 'all':
                raise ValueError('Błędny wartość parametru rows. Musi być to lista lub "all".')
        elif columns != 'all':
            raise ValueError('Błędny wartość parametru columns. Musi być to lista lub "all".')
        if wide_view:
            with pd.option_context('display.min_rows', nrow, 'display.max_columns', ncol, 'display.width', 1000):
                print(self.data.loc[display_rows, display_columns])
        else:
            print(self.data.loc[display_rows, display_columns])

    def summary(self):
        print(self.data.info())


class MetallurgyDataObject(DataObject):
    def __init__(self, file : File):
        super().__init__(file)

    # =============================================================
    # Proste wywalanie/uzupełnianie nulli
    # =============================================================
    def handle_nulls(self, columns, method='drop', value=0):
        if method == 'drop':
            self.data = self.data.dropna(subset=columns).reset_index(drop=True)
        elif method == 'fill_zeros':
            self.data.loc[:, columns] = self.data.loc[:, columns].fillna(0)
        elif method == 'correct':
            self.data.loc[:, columns] = self.data.loc[:, columns].fillna(value)
        else:
            raise ValueError('Nieznane metoda wypełniania została podana jako parametr do procedury.')

    # =============================================================
    # Dedykowane pod konkretne kolumny procedury uzupełniające dane
    # =============================================================

    def complement_sulfur_phosphorus(self):
        self.data.loc[self.data['sulfur'] == 0, 'sulfur'] = 0.003
        self.data.loc[self.data['phosphorus'] == 0, 'phosphorus'] = 0.003

    def complement_perlite_ferrite(self):
        def normalize(x, y):
            if x == 0 and y == 0:
                return [np.nan, np.nan]
            if x != 0 and y == 0:
                return [x / 100, 1 - x / 100]
            if x == 0 and y != 0:
                return [1 - y / 100, y / 100]
            else:
                return [x / (x + y), y / (x + y)]

        per_ferr = self.data[['perlite_perc', 'ferrite_perc']].fillna(0)
        per_ferr = per_ferr.apply(lambda x: normalize(x[0], x[1]), axis=1, result_type='expand').rename(columns={0: 'perlite_perc', 1: 'ferrite_perc'})

        self.data.loc[:, ['perlite_perc', 'ferrite_perc']] = per_ferr

    # Szukanie outlierow
    def removing_outliers(self, column, method='drop', scale=1):
        outliers = []
        print(len(list([column])))
        if len(list([column])) != 1:
            raise ValueError('Podaj tylko jedną kolumnę.')
        else:
            data_1 = self.data.loc[:, column]
            threshold = 3
            mean_1 = np.mean(data_1)
            std_1 = np.std(data_1)
            for y in data_1:
                z_score = (y - mean_1) / std_1
                if np.abs(z_score) > threshold:
                    outliers.append(y)
            print('Outliery dla', column, ': \n', outliers)

            if method == 'drop':
                self.data = self.data[~self.data.loc[:, column].isin(outliers)].reset_index(drop=True)
            elif method == 'correct':
                self.data[self.data.loc[:, column].isin(outliers)] = self.data[self.data.loc[:, column].isin(outliers)]*scale
            else:
                print('Nieznane metoda wypełniania została podana jako parametr do procedury. Tylko wypisanie outlierów.')

            return outliers

    def discretization(self, column, by):
        self.data.loc[self.data.loc[:, column] > by, column] = 'high'
        self.data.loc[self.data.loc[:, column] != 'high', column] = 'low'
        self.data.loc[:, column].astype('category')

        # Uzupełnienie temperatury - ustalone 21 stopni
    def complement_temperature(self):
        self.data['impact_strength_temp'] = self.data['impact_strength_temp'].fillna(21)

    # Na zewnętrznych danych (plus dane od Kochańskiego) wyuczyć model i wgrać go, aby dokonał prognozy na tym zbiorze danych.
    def fill_durability(self):
        pass

    def fill_tensile(self):
        pass

    # =============================================================
    # Dane ekstrachowane do innych modeli (np. w R)
    # =============================================================
    def extract_data_to_graphite_model(self):
        X = ['carbon', 'silicon', 'sulfur', 'phosphorus', 'magnesium', 'manganese', 'nickel', 'copper', 'molybdenum', 'chromium', 'aluminium', 'tin']
        Y = ['graphite_precipitation', 'spheroid_diameter', 'nodularity']
        export = self.data[X + Y].drop_duplicates(subset=X).dropna(subset=['graphite_precipitation'])
        export.to_csv('data/processed/graphite_model.csv', index=False)

    # TODO: Dopisać więcej metod...
    #   1) Przerobić zmienne odpowiedzi.
    #   2) Wydzielić dane (nie puste wiersze) do modelu prognozującego wytrzymałość w stanie lanym (durability_liquid_state) i odporność na rozciąganie (tensile_liquid_state).
    #   3) Zrobić w miare sensowne modele do predykcji tych dwóch cech. Wagi lub współczynniki zapisać do folderu models.

