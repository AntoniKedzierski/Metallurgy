from processing.reader import ExcelFile
from processing.processing import MetallurgyDataObject
# Hejo
if __name__ == '__main__':
    f = ExcelFile('data/raw/metallurgy.xlsx')
    f.read(usecols='K:AR,AT:BF', first_row=11)

    d = MetallurgyDataObject(f)
    d.refactor_columns_with_file('data/config/rename_dict_en.json')

    # coś robimy na ramce danych..
    d.handle_nulls(columns=['carbon', 'austenitization_temp', 'austenitization_duration', 'hardening_temp', 'hardening_duration'], method='drop')

    # ...
    d.handle_nulls(columns=['silicon', 'magnesium', 'manganese', 'nickel', 'copper', 'molybdenum', 'chromium', 'aluminium', 'tin'], method='fill_zeros')

    # ...
    d.complement_perlite_ferrite()

    # ...
    d.print(wide_view=True, nrow=400)

    d.summary()

