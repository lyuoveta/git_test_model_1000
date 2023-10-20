import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

# Open the CSV file
df = pd.read_csv('reviews_Biotin.csv')

# Remove rows with empty values in the third column
df = df.dropna(subset=['reviews'])
df = df.drop_duplicates()

# Remove leading and trailing white spaces in all columns
df = df.apply(lambda x: x.str.strip())

# Add new columns
df.insert(0, 'ID', range(1, len(df) + 1))
df['Classes'] = pd.Series([])

# Create a new workbook and get the active sheet
wb = Workbook()
ws = wb.active

# Save the DataFrame as an Excel file with delimiter
for r in dataframe_to_rows(df, index=False, header=True):
    ws.append(r)

# Set the freeze panes and auto filter
ws.freeze_panes = "A2"
ws.auto_filter.ref = ws.dimensions

# Save the Excel file
wb.save('Разметка_Biotin_12.xlsx')