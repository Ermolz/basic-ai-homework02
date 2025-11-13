import numpy as np
from sklearn.datasets import load_iris


# ===========================
# Допоміжні функції
# ===========================

def train_test_split_custom(X, y, test_size=0.3, random_state=42):
    """
    Ручна реалізація train/test split:
    - перемішує індекси
    - відокремлює частину вибірки під тест
    """
    rng = np.random.RandomState(random_state)
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    test_count = int(n_samples * test_size)
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def load_iris_from_csv(path="iris.csv"):
    """
        Завантажуємо датасет Iris з локального CSV-файлу.
        Очікується формат:
          sepal_length,sepal_width,petal_length,petal_width,target
        """
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    X = data[:, :-1]        # усі стовпчики, крім останнього — ознаки
    y = data[:, -1].astype(int)     # останній стовпчик — номер класу
    return X, y

def accuracy(y_true, y_pred):
    """
    Обчислення точності класифікації:
    частка об'єктів, для яких передбачений клас = справжньому.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return (y_true == y_pred).mean()


class StandardScalerManual:
    """
        Ручна реалізація StandardScaler:
        - запам'ятовує середнє та стандартне відхилення по кожній ознаці
        - масштабує ознаки до нульового середнього та одиничного стандартного відхилення
    """
    def fit(self, X):
        X = np.asarray(X)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0 # захист від ділення на нуль
        return self

    def transform(self, X):
        X = np.asarray(X)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ===========================
# 1) Метричний метод: k-NN
# ===========================

class KNNClassifier:
    """
        Метричний метод класифікації — k найближчих сусідів (k-Nearest Neighbors).
        - Об'єкти уявляються як точки в багатовимірному просторі ознак R^d.
        - Для нового об'єкта шукаємо k найближчих точок з навчальної вибірки
          (за евклідовою відстанню) і обираємо клас більшості.
    """
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        """
        "Навчання" для k-NN:
        просто зберігаємо навчальні об'єкти та їхні мітки.
        """
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        return self

    def _predict_one(self, x):
        """
            Передбачення класу для одного об'єкта:
            - рахуємо відстані до всіх навчальних точок
            - беремо k найближчих
            - голосування більшості по класах
        """
        # Евклідова відстань до всіх об'єктів train
        dists = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        idx = np.argsort(dists)[:self.k]
        labels, counts = np.unique(self.y_train[idx], return_counts=True)
        return labels[np.argmax(counts)]

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_one(x) for x in X])


# =========================================
# 2) Ймовірнісний метод: Gaussian Naive Bayes
# =========================================

class GaussianNBManual:
    """
    Наївний баєсівський класифікатор з нормальним (гаусовим) розподілом ознак.
    Ідея:
    - для кожного класу оцінюємо середнє та дисперсію кожної ознаки,
    - припускаємо, що ознаки незалежні (наївне припущення),
    - застосовуємо формулу Баєса і обираємо клас з максимальною апостеріорною ймовірністю.
    """
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        self.class_prior_ = {}  # апріорні ймовірності P(C)
        self.mean_ = {}         # середні значення ознак для кожного класу
        self.var_ = {}          # дисперсії ознак для кожного класу

        for c in self.classes_:
            X_c = X[y == c]
            self.class_prior_[c] = X_c.shape[0] / X.shape[0]
            self.mean_[c] = X_c.mean(axis=0)
            self.var_[c] = X_c.var(axis=0) + 1e-9  # невелике згладжування
        return self

    def _log_gaussian_prob(self, c, x):
        """
        Лог-щільність багатовимірного нормального розподілу
        (за припущенням незалежних координат) для класу c.
        """
        mean = self.mean_[c]
        var = self.var_[c]
        return -0.5 * np.sum(np.log(2 * np.pi * var) + ((x - mean) ** 2) / var)

    def _predict_one(self, x):
        """
        Обчислюємо лог-апостеріорні ймовірності для всіх класів
        і обираємо клас з найбільшим значенням.
        """
        best_class = None
        best_log_prob = -np.inf
        for c in self.classes_:
            log_prior = np.log(self.class_prior_[c])
            log_likelihood = self._log_gaussian_prob(c, x)
            log_posterior = log_prior + log_likelihood
            if log_posterior > best_log_prob:
                best_log_prob = log_posterior
                best_class = c
        return best_class

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_one(x) for x in X])


# ======================================
# 3) Дерево рішень (логічний / графовий метод)
# ======================================

class DecisionTreeNode:
    """
    Вузол дерева рішень:
    - gini: значення критерію Джині для цього вузла
    - num_samples: скільки об'єктів потрапляє в вузол
    - num_samples_per_class: розподіл об'єктів по класах
    - predicted_class: клас, який "перемагає" у вузлі
    - feature_index, threshold: ознака та поріг, за якими робимо розбиття
    - left, right: дочірні вузли
    """
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None


def gini_impurity(y):
    """
    Індекс Джині — міра "змішаності" класів у вузлі:
    G = 1 - sum(p_k^2), де p_k — частка класу k.
    Нуль → усі об'єкти одного класу.
    """
    m = len(y)
    if m == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / m
    return 1.0 - np.sum(p ** 2)


class DecisionTreeClassifierManual:
    """
    Класифікатор "дерево рішень":
    - представляє знання у вигляді дерева з логічними правилами виду:
        if feature_j <= threshold → ліве піддерево
        else → праве піддерево
    - використовує індекс Джині для пошуку найкращих розбиттів.
    """
    def __init__(self, max_depth=None, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features  # якщо None — беремо всі ознаки

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y, depth=0)
        return self

    def _best_split(self, X, y):
        """
        Пошук найкращого розбиття:
        - перебираємо ознаки (або випадкову підмножину ознак)
        - для кожної знаходимо можливі пороги
        - обираємо розбиття з мінімальним значенням Джині у дочірніх вузлах.
        """
        m, n_features = X.shape
        if m < self.min_samples_split:
            return None, None

        # які ознаки розглядати
        if self.max_features is None:
            feature_indices = range(n_features)
        else:
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)

        best_gini = 1.0
        best_idx, best_thr = None, None

        for idx in feature_indices:
            sorted_idx = np.argsort(X[:, idx])
            X_sorted = X[sorted_idx]
            y_sorted = y[sorted_idx]

            unique_values = np.unique(X_sorted[:, idx])
            if unique_values.shape[0] == 1:
                continue
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0

            for thr in thresholds:
                left_mask = X_sorted[:, idx] <= thr
                right_mask = ~left_mask

                y_left = y_sorted[left_mask]
                y_right = y_sorted[right_mask]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                gini_left = gini_impurity(y_left)
                gini_right = gini_impurity(y_right)
                gini = (len(y_left) * gini_left + len(y_right) * gini_right) / m

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = thr

        return best_idx, best_thr

    def _grow_tree(self, X, y, depth):
        """
        Рекурсивне побудування дерева:
        - створюємо вузол
        - якщо можна — знаходимо найкраще розбиття і ростимо ліве/праве піддерево
        - якщо ні — повертаємо лист.
        """
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = int(np.argmax(num_samples_per_class))
        node = DecisionTreeNode(
            gini=gini_impurity(y),
            num_samples=y.size,
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        # умови зупинки росту дерева
        if self.max_depth is not None and depth >= self.max_depth:
            return node
        if node.gini == 0.0:
            return node

        idx, thr = self._best_split(X, y)
        if idx is None:
            return node

        indices_left = X[:, idx] <= thr
        X_left, y_left = X[indices_left], y[indices_left]
        X_right, y_right = X[~indices_left], y[~indices_left]

        node.feature_index = idx
        node.threshold = thr
        node.left = self._grow_tree(X_left, y_left, depth + 1)
        node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _predict_one(self, x):
        """
        Обхід дерева для одного об'єкта:
        рухаємось зверху вниз, поки не дійдемо до листа,
        і повертаємо його predicted_class.
        """
        node = self.tree_
        while node.left is not None:
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_one(x) for x in X])


# ======================================
# 4) Логістична регресія (multiclass softmax)
# ======================================

class LogisticRegressionMulticlass:
    """
    Багатокласова логістична регресія (softmax-регресія).
    - Лінійна модель: scores = XW + b
    - Ймовірності класів: softmax(scores)
    - Навчання: градієнтний спуск по крос-ентропійній втраті
      з опціональною L2-регуляризацією.
    """
    def __init__(self, lr=0.1, n_epochs=1000, regularization=0.0):
        self.lr = lr
        self.n_epochs = n_epochs
        self.regularization = regularization

    def _softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        classes = np.unique(y)
        self.classes_ = classes
        n_classes = len(classes)

        # Кодуємо класи в one-hot вектори
        class_to_index = {c: idx for idx, c in enumerate(classes)}
        y_idx = np.array([class_to_index[c] for c in y])

        Y_onehot = np.zeros((n_samples, n_classes))
        Y_onehot[np.arange(n_samples), y_idx] = 1.0

        # Ініціалізація ваг
        rng = np.random.RandomState(42)
        self.W = rng.normal(scale=0.01, size=(n_features, n_classes))
        self.b = np.zeros(n_classes)

        # Градієнтний спуск
        for epoch in range(self.n_epochs):
            scores = X @ self.W + self.b
            probs = self._softmax(scores)

            error = probs - Y_onehot
            grad_W = (X.T @ error) / n_samples
            grad_b = np.mean(error, axis=0)

            if self.regularization > 0:
                grad_W += 2 * self.regularization * self.W

            self.W -= self.lr * grad_W
            self.b -= self.lr * grad_b

        return self

    def predict(self, X):
        X = np.asarray(X)
        scores = X @ self.W + self.b
        probs = self._softmax(scores)
        idxs = np.argmax(probs, axis=1)
        return self.classes_[idxs]


# ======================================
# 5) Метод опорних векторів (лінійний SVM, one-vs-rest)
# ======================================

class LinearSVMOneVsRest:
    """
    Лінійний SVM у постановці "one-vs-rest":
    - для кожного класу навчається окремий двокласовий SVM:
        позитивний клас = цей клас, негативний = всі інші
    - використовується hinge-втрата та L2-регуляризація.
    """
    def __init__(self, lr=0.001, n_epochs=2000, C=10.0):
        self.lr = lr
        self.n_epochs = n_epochs
        self.C = C  # коефіцієнт регуляризації (штраф за помилки)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Матриця ваг: по стовпчику на кожен клас
        self.W = np.zeros((n_features, n_classes))
        self.b = np.zeros(n_classes)

        for class_idx, c in enumerate(self.classes_):
            # Перетворюємо задачу на бінарну: цей клас = 1, інші = -1
            y_bin = np.where(y == c, 1, -1)
            w = np.zeros(n_features)
            b = 0.0

            for epoch in range(self.n_epochs):
                margins = y_bin * (X @ w + b)
                misclassified = margins < 1 # де порушена умова margin >= 1
                if not np.any(misclassified):
                    continue

                # Градієнти hinge-втрати + L2-регуляризація
                grad_w = w - self.C * (X[misclassified].T @ y_bin[misclassified]) / n_samples
                grad_b = -self.C * np.mean(y_bin[misclassified])

                w -= self.lr * grad_w
                b -= self.lr * grad_b

            self.W[:, class_idx] = w
            self.b[class_idx] = b

        return self

    def predict(self, X):
        X = np.asarray(X)
        scores = X @ self.W + self.b # кожен стовпець — оцінка для класу
        idxs = np.argmax(scores, axis=1)
        return self.classes_[idxs]


# ======================================
# 6) Random Forest (ансамбль дерев рішень)
# ======================================

class RandomForestClassifierManual:
    """
        Random Forest (випадковий ліс):
        - ансамбль з n_trees дерев рішень
        - кожне дерево навчається на випадковому бутстрап-зразку з даних
          і, за бажанням, на випадковій підмножині ознак
        - фінальне передбачення: голосування більшості по деревам.
        """
    def __init__(self, n_trees=20, max_depth=None, min_samples_split=2, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape

        # Якщо кількість ознак не задано — беремо sqrt(d), як у класичному Random Forest
        if self.max_features is None:
            self.max_features_ = int(np.sqrt(n_features))
        else:
            self.max_features_ = self.max_features

        self.trees_ = []
        rng = np.random.RandomState(42)
        for i in range(self.n_trees):
            # Бутстрап: вибираємо з поверненням n_samples об'єктів
            indices = rng.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            tree = DecisionTreeClassifierManual(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features_
            )
            tree.fit(X_sample, y_sample)
            self.trees_.append(tree)

        return self

    def predict(self, X):
        X = np.asarray(X)
        # Збираємо передбачення всіх дерев (shape: n_trees x n_samples)
        all_preds = np.array([tree.predict(X) for tree in self.trees_])
        all_preds = all_preds.T  # (n_samples, n_trees)

        # Для кожного об'єкта — голосування більшості по класах
        y_pred = []
        for sample_preds in all_preds:
            labels, counts = np.unique(sample_preds, return_counts=True)
            y_pred.append(labels[np.argmax(counts)])
        return np.array(y_pred)


# ======================================
# Запуск і порівняння всіх методів на Iris
# ======================================

if __name__ == "__main__":
    # 1. Завантажуємо локальний датасет Iris з CSV
    X, y = load_iris_from_csv("iris.csv")

    # 2. Ділимо на навчальну та тестову вибірки
    X_train, X_test, y_train, y_test = train_test_split_custom(
        X, y, test_size=0.3, random_state=42
    )

    # 3. Масштабуємо ознаки для методів, чутливих до масштабу (k-NN, логістична регресія, SVM)
    scaler = StandardScalerManual()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Опис моделей:
    #   - metric_kNN              → метричний метод (k найближчих сусідів)
    #   - probabilistic_GaussianNB→ ймовірнісний метод (наївний Баєс)
    #   - logical_DecisionTree    → логічний/графовий метод (дерево рішень)
    #   - logistic_regression     → класична логістична регресія
    #   - modern_SVM              → метод опорних векторів (новітній метод)
    #   - modern_RandomForest     → ансамблевий метод (рандомний ліс)
    models = {
        "metric_kNN": KNNClassifier(k=5),
        "probabilistic_GaussianNB": GaussianNBManual(),
        "logical_DecisionTree": DecisionTreeClassifierManual(),
        "logistic_regression": LogisticRegressionMulticlass(
            lr=0.1, n_epochs=1000, regularization=0.01
        ),
        "modern_SVM": LinearSVMOneVsRest(
            lr=0.001, n_epochs=2000, C=10.0
        ),
        "modern_RandomForest": RandomForestClassifierManual(
            n_trees=20, max_depth=None
        ),
    }

    # 5. Навчання і оцінка всіх методів на одному й тому ж датасеті
    for name, model in models.items():
        print("=" * 80)
        print(f"Метод: {name}")

        # Методи, які потребують масштабування ознак
        if isinstance(model, (KNNClassifier, LogisticRegressionMulticlass, LinearSVMOneVsRest)):
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        acc = accuracy(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
