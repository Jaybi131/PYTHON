import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import optuna
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder

#Загрузка данных:
file_path = "students.csv"
df = pd.read_csv(file_path)

# Первичный анализ данных
print("Erste fünf Zeilen des Datensatzes:\n", df.head())
print("\nInformationen über den Datensatz:\n")
df.info()
print("\nBeschreibung der numerischen Merkmale:\n", df.describe())
print("\nÜberblick über die Verteilung der Werte in den kategorischen Merkmalen:\n", df.describe(include=['object']))

# Получение основных характеристик данных
merkmale = df.columns
anzahl_datenreihen = df.shape[0]

print("Merkmale:\n", merkmale)
print("\nAnzahl der Datenreihen:", anzahl_datenreihen)

# Определение типов характеристик
merkmale_typen = df.dtypes
kategorische_merkmale = df.select_dtypes(include=['object']).columns
numerische_merkmale = df.select_dtypes(include=['number']).columns

print("\nTypen der Merkmale:\n", merkmale_typen)
print("\nKategorische Merkmale:\n", kategorische_merkmale)
print("\nNumerische Merkmale:\n", numerische_merkmale)

# Идентификация шкал измерений
skalen = {}
for merkmal in merkmale:
    if merkmal in kategorische_merkmale:
        unique_values = df[merkmal].unique()
        if len(unique_values) < 10:
            skala = "nominal" if len(set(unique_values)) == len(unique_values) else "ordinal"
        else:
            skala = "nominal"
    elif merkmal in numerische_merkmale:
        skala = "verhältnis"  # Annahme: numerische Daten haben einen natürlichen Nullpunkt
    else:
        skala = "unbekannt"
    skalen[merkmal] = skala
    print(f"{merkmal}: {skala}")

# Поиск пропущенных значений
fehlende_werte = df.isnull().sum()

print("\nFehlende Werte pro Merkmal:\n", fehlende_werte)

# Проверка нулевых значений в столбцах с оценками
print("\nAnzahl der MathScore-Werte, die 0 sind:", (df["MathScore"] == 0).sum())
print("Anzahl der ReadingScore-Werte, die 0 sind:", (df["ReadingScore"] == 0).sum())
print("Anzahl der WritingScore-Werte, die 0 sind:", (df["WritingScore"] == 0).sum())

# 1. Сколько мужчин и сколько женщин включены в набор данных?
gender_counts = df['Gender'].value_counts()
print(gender_counts)

# Построение столбчатой диаграммы для количества мужчин и женщин
sns.countplot(data=df, x='Gender')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# 2. Коробчатые диаграммы для оценки влияния пола на результаты тестов
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

sns.boxplot(ax=axes[0], data=df, x='Gender', y='MathScore')
axes[0].set_title('Math Score by Gender')

sns.boxplot(ax=axes[1], data=df, x='Gender', y='ReadingScore')
axes[1].set_title('Reading Score by Gender')

sns.boxplot(ax=axes[2], data=df, x='Gender', y='WritingScore')
axes[2].set_title('Writing Score by Gender')

plt.tight_layout()
plt.show()

# Расчет средних значений
mean_scores = df.groupby('Gender')[['MathScore', 'ReadingScore', 'WritingScore']].mean()
print(mean_scores)

# 3. Функция для вычисления выбросов
def calculate_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return ((data < lower_bound) | (data > upper_bound)).sum()

# Применение функции к каждому тесту для каждого пола
outliers = df.groupby('Gender')[['MathScore', 'ReadingScore', 'WritingScore']].apply(calculate_outliers)
print(outliers)

# Построение и визуализация корреляционной (Korrelationsmatrix) матрицы
korrelationsmatrix = df[['MathScore', 'ReadingScore', 'WritingScore']].corr()

print("Korrelationsmatrix:\n", korrelationsmatrix)


plt.figure(figsize=(10, 7))
sns.heatmap(korrelationsmatrix, annot=True, cmap='coolwarm', center=0)
plt.title('Korrelationsmatrix der Testergebnisse')
plt.show()
# (LabelEncoder) Кодирование категориальных данных и построение коробчатых диаграмм
label_encoder = LabelEncoder()
df['PracticeSport_encoded'] = label_encoder.fit_transform(df['PracticeSport'])

# Boxplots
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.boxplot(x='PracticeSport_encoded', y='MathScore', data=df)
plt.xlabel('Practice Sport')
plt.ylabel('Math Score')
plt.title('Mathematiknoten nach Übungshäufigkeit')
plt.subplot(1, 3, 2)
sns.boxplot(x='PracticeSport_encoded', y='ReadingScore', data=df)
plt.xlabel('Practice Sport')
plt.ylabel('Reading Score')
plt.title('Lesenoten nach Übungshäufigkeit')

plt.subplot(1, 3, 3)
sns.boxplot(x='PracticeSport_encoded', y='WritingScore', data=df)
plt.xlabel('Practice Sport')
plt.ylabel('Writing Score')
plt.title('Schreibenoten nach Übungshäufigkeit')

plt.tight_layout()
plt.show()

# Анализ влияния образовательного уровня родителей на результаты тестов
mittelwerte = df.groupby('ParentEduc')[['MathScore', 'ReadingScore', 'WritingScore']].mean().reset_index()

mittelwerte.plot(x='ParentEduc', kind='bar', figsize=(12, 6))
plt.title('Durchschnittliche Testergebnisse nach Bildungsniveau der Eltern')
plt.ylabel('Durchschnittliche Punktzahl')
plt.xlabel('Bildungsniveau der Eltern')
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.show()

# Анализ влияния количества недельных часов обучения на результаты тестов
df['WklyStudyHours'] = df['WklyStudyHours'].replace({
    '< 5': '0-5',
    '5 - 10': '5-10',
    '10 - 20': '10-20',
    '20 - 30': '20-30',
    '30 - 40': '30-40',
    '> 40': '40+'
})

mittelwerte_studienstunden = df.groupby('WklyStudyHours')[['MathScore', 'ReadingScore', 'WritingScore']].mean().reset_index()

mittelwerte_studienstunden.plot(x='WklyStudyHours', kind='bar', figsize=(12, 6))
plt.title('Durchschnittliche Testergebnisse nach wöchentlichen Studienstunden')
plt.ylabel('Durchschnittliche Punktzahl')
plt.xlabel('Wöchentliche Studienstunden')
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.show()

# Загрузка данных


# Сохраняем оригинальные данные для анализа позже
df_original = df.copy()

# One-Hot-Encoding для категориальных признаков
categorical_features = df.select_dtypes(include=['object']).columns
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Импьютация пропущенных значений с использованием kNN
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df_encoded), columns=df_encoded.columns)

# Проверка на наличие пропущенных значений после импьютации
missing_values_after_imputation = df_imputed.isnull().sum()
print("ПFehlende Werte nach Imputation:\n", missing_values_after_imputation)

# Масштабирование данных
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_imputed)

# Определение функции для Optuna
def objective(trial):
    n_clusters = trial.suggest_int('n_clusters', 2, 10)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(df_scaled)
    score = silhouette_score(df_scaled, kmeans.labels_)
    return score

# Создание и оптимизация Optuna-исследования
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

# Оптимальное количество кластеров
optimal_clusters = study.best_params['n_clusters']
print(f"Optimal Anzahl der Clusters: {optimal_clusters}")

# Кластеризация с оптимальным количеством кластеров
kmeans = KMeans(n_clusters=optimal_clusters, random_state=0)
df_imputed['Cluster'] = kmeans.fit_predict(df_scaled)

# PCA для уменьшения размерности
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# Результаты PCA в DataFrame
df_pca = pd.DataFrame(df_pca, columns=['PCA1', 'PCA2'])
df_pca['Cluster'] = df_imputed['Cluster']

# Визуализация кластеров
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='Cluster', palette='Set1')
plt.title('Visualisierung mit PCA')
plt.show()

# Кластеризация с 3 кластерами
kmeans_3 = KMeans(n_clusters=3, random_state=0)
df_imputed['Cluster_3'] = kmeans_3.fit_predict(df_scaled)

# Добавляем колонку Gender обратно в df_imputed
df_imputed['Gender'] = df_original['Gender']

# 3D-плот
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df['MathScore'], df['ReadingScore'], df['WritingScore'], c=df_imputed['Cluster_3'], cmap='Set1')

ax.set_xlabel('Math Score')
ax.set_ylabel('Reading Score')
ax.set_zlabel('Writing Score')
plt.title('3D-Plot der Testergebnisse nach Cluster')
plt.legend(*scatter.legend_elements(), title='Кластер')
plt.show()

# Распределение по полу в 3 кластерах
gender_distribution = df_imputed.groupby('Cluster_3')['Gender'].value_counts(normalize=True).unstack()
print("Geschlechterverteilung in 3 Clustern:\n", gender_distribution)

# Визуализация распределения по полу
gender_distribution.plot(kind='bar', stacked=True, figsize=(10, 7))
plt.title('Geschlechterverteilung in 3 Clustern')
plt.xlabel('Cluster')
plt.ylabel('Teil')
plt.show()

