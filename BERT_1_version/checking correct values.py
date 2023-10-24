import pandas as pd
import matplotlib.pyplot as plt

# Чтение CSV-файла
df = pd.read_csv("tagged_data.csv")

# deleting additional information
df = df.drop([df.columns[0], df.columns[2], df.columns[3],df.columns[4]], axis=1)

# вывод строк с некорректными значениями
row_numbers = []
def is_neutral(row):
    values = row.iloc[2:32]  # Пример, где 3-34 это диапазон столбцов, которые нужно проверить
    if all(val == 0 or val == 1 for val in values):
        return False
    return True

neutral_rows = []
for index, row in df.iterrows():
    if is_neutral(row):
        neutral_rows.append(index)

for row_number in neutral_rows:
    print(f"Строка {row_number} не относится ни к корректным, ни к некорректным значениям.")

# вывод строк с пропущенными значенями
zero_rows = df[(df.iloc[:, 30] == 0) & (df.iloc[:, 31] == 0)]
print(zero_rows.index)
row_index = 19
print(df.loc[row_index])

# вывод корректных и некорректных отзывов
count_ones = len(df[df.iloc[:, 30] == 1])
count_ones_2 = len(df[df.iloc[:, 31] == 1])
column_name = df.columns[30]
column_name_2 = df.columns[31]
print(column_name, count_ones, column_name_2, count_ones_2)

# вывод количества строк
numrows = len(df)
print(numrows)

# исправление некорректных значений

