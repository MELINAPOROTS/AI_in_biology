# Импортируем все необходимые библиотеки
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Определение функции classification_training
def classification_training(data):
    # Преобразуем столбец 'mental_wellness_index_0_100' в бинарную целевую переменную
    # Значения меньше 15 -> 0, больше или равны 15 -> 1
    y = (data['mental_wellness_index_0_100'] >= 15).astype(int)

    # Выбираем все столбцы, кроме целевого ('mental_wellness_index_0_100') и идентификатора ('user_id')
    feature_columns = [col for col in data.columns if col not in ['mental_wellness_index_0_100', 'user_id']]
    X = data[feature_columns].copy()

    # Обработка категориальных признаков
    # Используем LabelEncoder для преобразования строковых значений в числа
    categorical_features = ['gender', 'occupation', 'work_mode']
    le_dict = {}
    for feature in categorical_features:
        if feature in X.columns:
            le = LabelEncoder()
            X.loc[:, feature] = le.fit_transform(X[feature].astype(str))
            le_dict[feature] = le 

    # Разделение данных на обучающую и тестовую выборки
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        shuffle=True,
        random_state=37 
    )

    # Обучение модели K-ближайших соседей (KNN)
    # Масштабируем данные, так как KNN чувствителен к масштабу
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    knn_model = KNeighborsClassifier(
        n_neighbors=5  #Малое значение (например, 1 или 2) может привести к переобучению, так как модель будет слишком подстраиваться под обучающие данные. Большое значение (например, 100) может привести к недообучению, так как модель будет слишком обобщённой. 5 — это компромиссное значение, которое позволяет модели быть достаточно чувствительной к локальным особенностям данных, но при этом не переобучаться на шуме.
    )
    knn_model.fit(X_train_scaled, y_train) 

    # Обучение модели решающего дерева
    # Ограничиваем глубину и количество образцов для разбиения, чтобы избежать переобучения
    dt_model = DecisionTreeClassifier(
        max_depth=7, # Ограничиваем глубину дерева. Ограничение глубины дерева предотвращает его чрезмерное разрастание. Если дерево слишком глубокое, оно может запомнить обучающую выборку (переобучиться). Глубина 7 — это достаточная сложность, чтобы захватить важные закономерности, но не слишком большая, чтобы избежать переобучения.
        min_samples_split=10, # Минимальное количество образцов для разбиения внуттреннего узла. Если в узле меньше 10 образцов, разбиение не будет происходить. Это предотвращает создание слишком специфичных правил на основе небольшого количества данных, что уменьшает переобучение.
        min_samples_leaf=5, # Минимальное количество образцов в листе. Если разбиение приведёт к созданию листа с меньшим количеством образцов, оно не выполнится. Это увеличивает обобщающую способность модели, уменьшая чувствительность к шуму в данных.
        random_state=37 
    )
    dt_model.fit(X_train_raw, y_train)

    # Предсказания и вычисление метрик
    # Для KNN используем масштабированные данные
    y_pred_knn = knn_model.predict(X_test_scaled)
    knn_accuracy = accuracy_score(y_test, y_pred_knn)
    knn_f1 = f1_score(y_test, y_pred_knn)

    # Предсказания и оценка для решающего дерева
    # Для дерева используем необработанные данные
    y_pred_dt = dt_model.predict(X_test_raw) 
    dt_accuracy = accuracy_score(y_test, y_pred_dt)
    dt_f1 = f1_score(y_test, y_pred_dt)

    # Вывод метрик в заданном формате
    print(f"KNN: {knn_accuracy:.4f}; {knn_f1:.4f}")
    print(f"DT: {dt_accuracy:.4f}; {dt_f1:.4f}")

    # Построение дерева решений
    plt.figure(figsize=(30, 15))
    plot_tree(dt_model, feature_names=X.columns, filled=True, rounded=True, fontsize=10)
    plt.show() 

#data = pd.read_csv("processed_DB_3.csv") 
#classification_training(data)

