import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

from processing.reader import ExcelFile
from processing.processing import MetallurgyDataObject

if __name__ == '__main__':
    f = ExcelFile('data/raw/metallurgy.xlsx')
    f.read(usecols='K:BF', first_row=11)

    d = MetallurgyDataObject(f)
    d.refactor_columns_with_file('data/config/rename_dict_en.json')

    # coś robimy na ramce danych..
    #'carbon', 'silicon'
    d.handle_nulls(columns=['carbon', 'silicon'], method='drop')

    # "sulfur" 'phosphorus'
    d.complement_sulfur_phosphorus()
    d.handle_nulls(columns=['sulfur', 'phosphorus'], method='correct', value=0.004)

    # 'magnesium'
    d.removing_outliers('magnesium', method='correct', scale=0.1)
    d.handle_nulls(columns=['magnesium', 'manganese'], method='correct', value=0.03)

    # 'manganese' - bimodalny rozkład
    # d.discretization(column='manganese', by=0.25) # NA będą low
    # d.print(wide_view=True, nrow=400, columns=['manganese'], rows=list(range(1160, 1190))) #test metody

    d.handle_nulls(columns=['nickel', 'copper', 'molybdenum', 'chromium', 'aluminium', 'tin'], method='fill_zeros')
    d.handle_nulls(columns=['austenitization_temp', 'austenitization_duration', 'hardening_temp', 'hardening_duration'], method='drop')
    d.drop_columns(columns=['graphite_precipitation', 'graphite_precipitation_perc', 'spheroid_diameter', 'spheroid_size', 'nodularity'])

    # NA to do:
    # 'austenitization_temp', 'austenitization_duration', 'hardening_temp', 'hardening_duration'
    # ...
    d.complement_perlite_ferrite()
    d.complement_temperature()

    d.fill_durability()
    d.fill_tensile()
    d.export()




