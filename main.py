import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from processing.reader import ExcelFile
from processing.processing import MetallurgyDataObject

if __name__ == '__main__':
    f = ExcelFile('data/raw/metallurgy.xlsx')
    f.read(usecols='K:AM,AT:AY, BA:BF', first_row=11) #Wyrzucam Kolumny AN-AS oraz AZ (mają zero wartości)

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

    # Test metody complement_temperature()
    d.print(wide_view=True, nrow=400, columns=['impact_strength_temp'], rows=list(range(1160, 1190)))
    d.complement_temperature()
    d.print(wide_view=True, nrow=400, columns=['impact_strength_temp'], rows=list(range(1160, 1190)))

    # Kinga: analiza kolumn

    # Węgiel
    dane = f.data
    carbon = pd.DataFrame(dane['C [%]'])
    print(carbon)


    carbon.isna().sum()/carbon.count()*100 # Braki danych 4.6495% - mało
    plt.boxplot(dane['C [%]'].dropna())
    plt.show() # Wygląda spoko




