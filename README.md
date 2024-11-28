# Трек 1. Хаб: объединение данных пользователя в золотую запись

## Обработка данных:
Сначала данные загружаются из CSV файла. После загрузки из DataFrame мы игнорируем столбцы с монотонными значениями, поскольку они не добавляют полезной информации для анализа. Полные дубликаты строк также исключаются.

Далее применяются регулярные выражения для удаления из строк нежелательных символов, оставляя только буквы, дефисы и пробелы. Это важно для очистки столбцов с именами клиентов, чтобы обеспечить последовательность и чистоту данных.

Функция source_priority назначает числовые приоритеты различным источникам данных на основе заранее определенного словаря, что помогает в предпочтительном выборе данных. Функция get_client_bd вычисляет возраст клиентов на основе даты рождения, при этом исключая данные с ошибками, такие как будущие даты или возраст более 100 лет.

Функция preprocess_data преобразует даты в формат datetime, игнорируя некорректные значения (не даты), которые заменяются на NaT. Она также рассчитывает и добавляет столбец с возрастом клиента (client_yo), заполняя пропуски средним возрастом, чтобы сгладить выбросы данных. Для текстовых данных проводится стандартизация: они приводятся к нижнему регистру и очищаются от лишних пробелов.

Для выявления дубликатов данные кодируются в числовую форму с помощью LabelEncoder, что позволяет алгоритмам машинного обучения работать с категориальными признаками. Затем применяется кластеризация с использованием алгоритма DBSCAN, который выделяет возможные дубликаты. DBSCAN способен определять кластеры любой формы и выявляет записи, не принадлежащие к ни одному кластеру, как шум.

В результате метки кластеров сохраняются в новом столбце cluster, что позволяет легко идентифицировать группы дубликатов. После этого фильтруются записи, не попавшие в кластеры (обозначенные меткой -1), оставляя только те, которые принадлежат каким-либо кластерам.

Затем применяется функция select_golden_record, которая реализует логику выбора "золотой записи" из каждой группы дубликатов (кластера). "Золотая запись" представляет собой наиболее полную и качественную версию среди записей кластера. В ней рассчитывается полнота записи (completeness) на основе количества незаполненных значений, отбираются  лучшие записи по критериям: полнота, приоритет источника (source_priority) и дата обновления (update_date) для каждого кластера. Начальная "золотая запись" берется как первая из топовых записей, но затем она дополняется данными из других записей группы, если у них есть не пустые значения, отсутствующие в текущей "золотой записи".

После фильтрации ненужных колонок, используемых только для лучшей обработки, "золотые записи" сохраняются в новый CSV файл.

## Достигнутые результаты:

1. Создана программа с корректно работающими: загрузкой датасета, предобработкой датасета, кластеризацией, выделением «золотой записи» для каждого кластера, выдачей результата в виде csv-файла.
1. Обработка данных происходит с учётом их актуальности, частоты и полноты. Методология работы с данными описана в текстовом файле, находящемся в github-репозитории проекта.
1. Использована актуальная версия python 3.12.1 и современные библиотеки для работы с данными и машинного обучения (Pandas, Scikit-learn). Нет зависимости от внешних сервисов, т.к. использованы только open-source библиотеки.
1. Масштабируемость достигается разделением предобработки данных и группировки уже предобработанных данных по кластерам. Кроме того возможно использование библиотек, позволяющих параллельно обрабатывать данные для увеличения скорости работы.
