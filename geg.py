import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
import numpy as np
from dateutil.relativedelta import relativedelta
import re

print('Usage: <file_address>/<input_file_name> <file_address>/<output_file_name>')
input_file = input('input_file_name: ')
export_file = input('output_file_name: ')
data = pd.read_csv(input_file, low_memory=False)

data = data[[c for c
        in list(data)
        if len(data[c].unique()) > 1]]
data.drop_duplicates(inplace=True)


REGEXP_SYMBOLS = r'[^a-zA-Zа-яА-Я\-\ё\Ё\ ]'


def source_priority(source: str):
    SOURCE_PRIORITY = {'bank': 6, 'stream': 0, 'telco': 4, 'bank2': 5, 'fitness': 1, 'isp': 3, 'travel': 2}
    return SOURCE_PRIORITY.get(source.lower() if isinstance(source, str) else source, -1)

def get_client_bd(date):
    if not pd.isnull(date):
        r = relativedelta(pd.to_datetime('today'), date).years
        if r > 100 or r < 0:
            return None
        return int(relativedelta(pd.to_datetime('today'), date).years)
    return None



def preprocess_data(df: pd.DataFrame):
    df['source_priority'] = df['source_cd'].apply(source_priority)
 # Преобразование столбца 'update_date' в формат datetime и удаление некорректных дат
    df['create_date'] = pd.to_datetime(df['update_date'], errors='coerce')
    df['update_date'] = pd.to_datetime(df['update_date'], errors='coerce')
    df['client_bday'] = pd.to_datetime(df['client_bday'], errors='coerce')
    df['client_yo'] = df['client_bday'].apply(get_client_bd)
    mean_age = int(df['client_yo'].mean())

    df['client_yo'] = df['client_yo'].fillna(mean_age)

    # Очистка и стандартизация строковых данных
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'client_yo' and col != 'source_priority':
            df[col] = df[col].str.lower().fillna('').str.strip()  # Приведение к нижнему регистру, удаление пробелов

    # Очистка имени от чисел и мусорных символов
    df['client_first_name'] = df['client_first_name'].apply(lambda x: re.sub(REGEXP_SYMBOLS, '', x))
    # Очистка имени от чисел и мусорных символов
    df['client_middle_name'] = df['client_middle_name'].apply(lambda x: re.sub(REGEXP_SYMBOLS, '', x))
    # Очистка имени от чисел и мусорных символов
    df['client_last_name'] = df['client_last_name'].apply(lambda x: re.sub(REGEXP_SYMBOLS, '', x))
    # Очистка имени от чисел и мусорных символов
    df['client_fio_full'] = df['client_fio_full'].apply(lambda x: re.sub(REGEXP_SYMBOLS, '', x))

    return df


# Предбработка данных
data = preprocess_data(data)

GROUP_COLUMNS = ['client_fio_full', 'client_yo', 'client_cityzen', 'addr_region', 'addr_country', 'addr_city',
       'fin_rating', 'fin_loan_percent', 'stream_favorite_show', 'source_cd']


def find_duplicates(df):
    # Используем LabelEncoder для текстовых колонок
    label_encoders = {}

    # Обработка колонок
    for col in GROUP_COLUMNS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Создаем массив числовых фич для кластеризации
    features = df[GROUP_COLUMNS].values

    # Используем DBSCAN для поиска потенциальных дубликатов
    clustering = DBSCAN(eps=0.5, min_samples=2, metric='euclidean').fit(features)
    df['cluster'] = clustering.labels_

    # Обратное преобразование закодированных значений
    for col in GROUP_COLUMNS:
        le = label_encoders[col]
        df[col] = le.inverse_transform(df[col])

    return df


# Применение метода кластеризации
data = find_duplicates(data)

# Функция для выбора "золотой записи"
def select_golden_record(group: pd.DataFrame):
    # Пример: используем максимальную дату обновления и полноту полей как критерии
    group['completeness'] = group.notna().sum(axis=1)

    records_count = 3
    records = group.nlargest(records_count, ['source_priority', 'update_date', 'completeness'])

    golden_record = records.iloc[0]

    def is_valid_bday(bday: str):
        r = relativedelta(pd.to_datetime('today'), bday).years
        if r > 100 or r < 0:
            return False
        return True

    columns = records.columns
    for i, column in enumerate(columns):
        column_records = records[column].dropna()

        if column == 'client_bday':
            column_records = column_records[column_records.apply(is_valid_bday)]

        modes = column_records.mode()
        if len(modes) == 0:
            continue
        
        golden_record.iloc[i] = modes[0]

    return golden_record


# Группировка по кластерам и выбор золотой записи
golden_records = data.groupby('cluster', group_keys=False).apply(select_golden_record).reset_index(drop=True)

golden_records =  golden_records.drop(columns=['cluster', 'completeness', 'source_priority', 'client_yo'])

# Сохранение золотых записей в новый файл
golden_records.to_csv(export_file, index=False)

