import pandas as pd

excel_files = ['Разметка_Biotin.xlsx', 'Разметка_Berberine.xlsx',
               'Разметка_Colloidal_silver.xlsx','Разметка_Folic_acid.xlsx',
               'Разметка_Ginkgo.xlsx', 'Разметка_Krill_oil.xlsx',
               'Разметка_Melatonin.xlsx','Разметка_Vitamin_D.xlsx']

combined_df = pd.DataFrame()

for file in excel_files:

    df = pd.read_excel(file)
    table_name = file.split('.xlsx')[0]
    df['Table Name'] = table_name
    df = df.set_index('Table Name').reset_index()

    for col in df.columns:
        if col != 'Table Name' and col != 'dates' and col != 'details' and col != 'scores' and col != 'reviews':
            df[col] = df[col].astype(int)
        else:  df[col] = df[col].astype(str)

    combined_df = pd.concat([combined_df, df], ignore_index=True)

combined_df['ID'] = range(1, len(combined_df) + 1)

combined_df.to_csv('tagged_data.csv', index=False)