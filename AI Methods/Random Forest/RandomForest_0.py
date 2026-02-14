import random
import matplotlib.pyplot as plt

# =========================================================
# Utilities
# =========================================================
def mean(values):
    return sum(values) / len(values) if values else 0.0

def variance(values):
    if not values:
        return 0.0
    m = mean(values)
    return sum((v - m) ** 2 for v in values) / len(values)

def mse_of_split(left_y, right_y):
    n = len(left_y) + len(right_y)
    if n == 0:
        return float("inf")
    return (len(left_y) / n) * variance(left_y) + (len(right_y) / n) * variance(right_y)

# =========================================================
# CART-like Regression Tree (pure python)
# =========================================================
class TreeNode:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTreeRegressorPure:
    def __init__(self, max_depth=6, min_samples_split=8, max_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None
        self._rng = random.Random(random_state)

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)
        return self

    def predict_one(self, x):
        node = self.root
        while not node.is_leaf():
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def _best_split(self, X, y):
        n_features = len(X[0])

        # random subset of features
        feat_count = n_features if self.max_features is None else min(self.max_features, n_features)
        features = list(range(n_features))
        self._rng.shuffle(features)
        features = features[:feat_count]

        best_feature = None
        best_threshold = None
        best_score = float("inf")

        for f in features:
            vals = sorted(set(row[f] for row in X))
            if len(vals) <= 1:
                continue

            thresholds = [(vals[i] + vals[i+1]) / 2.0 for i in range(len(vals) - 1)]
            for thr in thresholds:
                left_y, right_y = [], []
                for row, target in zip(X, y):
                    (left_y if row[f] <= thr else right_y).append(target)

                if not left_y or not right_y:
                    continue

                score = mse_of_split(left_y, right_y)
                if score < best_score:
                    best_score = score
                    best_feature = f
                    best_threshold = thr

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth):
        if (depth >= self.max_depth) or (len(y) < self.min_samples_split) or (variance(y) < 1e-9):
            return TreeNode(value=mean(y))

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return TreeNode(value=mean(y))

        left_X, left_y, right_X, right_y = [], [], [], []
        for row, target in zip(X, y):
            if row[feature] <= threshold:
                left_X.append(row); left_y.append(target)
            else:
                right_X.append(row); right_y.append(target)

        if not left_y or not right_y:
            return TreeNode(value=mean(y))

        left_node = self._build_tree(left_X, left_y, depth + 1)
        right_node = self._build_tree(right_X, right_y, depth + 1)
        return TreeNode(feature=feature, threshold=threshold, left=left_node, right=right_node)

# =========================================================
# Random Forest Regressor (bagging + random features)
# =========================================================
class RandomForestRegressorPure:
    def __init__(self, n_estimators=80, max_depth=6, min_samples_split=6,
                 max_features=3, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self._rng = random.Random(random_state)
        self.trees = []

    def _bootstrap(self, X, y):
        n = len(y)
        Xb, yb = [], []
        for _ in range(n):
            i = self._rng.randrange(n)
            Xb.append(X[i])
            yb.append(y[i])
        return Xb, yb

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            Xb, yb = self._bootstrap(X, y)
            tree = DecisionTreeRegressorPure(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=self._rng.randrange(10**9)
            )
            tree.fit(Xb, yb)
            self.trees.append(tree)
        return self

    def predict_one(self, x):
        return mean([t.predict_one(x) for t in self.trees])

    def predict(self, X):
        return [self.predict_one(x) for x in X]

# =========================================================
# Time series -> supervised (lag features) + recursive forecast
# =========================================================
def make_lags(series, lags):
    X, Y = [], []
    for t in range(lags, len(series)):
        X.append([series[t - i] for i in range(1, lags + 1)])
        Y.append(series[t])
    return X, Y

def recursive_forecast(model, history, lags, h):
    hist = list(history)
    out = []
    for _ in range(h):
        x = [hist[-i] for i in range(1, lags + 1)]
        yhat = model.predict_one(x)
        out.append(yhat)
        hist.append(yhat)
    return out

# =========================================================
# Demo on your series + plotting
# =========================================================
if __name__ == "__main__":
    series = [
        280.05, 305.18, 324.14, 355.64, 394.53, 438.96, 501.13, 572.49, 635.81,
        726.68, 784.94, 727.34, 713.08, 747.34, 771.44, 894.97, 894.46, 968.30,
        1012.44, 1046.07, 1132.78, 1252.69, 1187.70, 1205.45, 1241.29, 1319.69,
        1426.17, 1856.83, 2023.56, 2129.56, 2239.85, 2389.03, 2660.00, 2770.00,
        2863.00, 3228.00, 3557.00, 3894.00, 4171.00, 4580.00, 4644.00, 4621.00,
        5142.00, 6440.00, 6448.00, 5960.00, 6155.00, 6507.00, 6001.00, 5927.00,
        6731.00, 7614.00, 7103.00, 6746.00, 7328.00, 8212.00, 7993.00, 8300.00,
        8731.00, 8702.00, 6677.00
    ]

    lags = 6
    test_size = 8
    horizon = 8

    X, y = make_lags(series, lags)

    # split (note: X/y start at t=lags+1)
    X_train, y_train = X[:-test_size], y[:-test_size]
    X_test, y_test = X[-test_size:], y[-test_size:]

    rf = RandomForestRegressorPure(
        n_estimators=80,
        max_depth=6,
        min_samples_split=6,
        max_features=3,
        random_state=42
    )
    rf.fit(X_train, y_train)

    test_preds = rf.predict(X_test)
    future = recursive_forecast(rf, series, lags, horizon)

    # --- Build time axes for plotting ---
    n = len(series)
    t_obs = list(range(1, n + 1))

    # test predictions correspond to last "test_size" targets in y
    # y indices align with original series at t = lags+1 ... n
    # so the last test_size predictions correspond to times:
    start_test_t = (lags + 1) + (len(y) - test_size)
    t_test = list(range(start_test_t, start_test_t + test_size))

    t_future = list(range(n + 1, n + horizon + 1))

    # --- Plot ---

#    plt.figure(figsize=(11, 5))
#    plt.plot(t_obs, series, label="Observed")
#    plt.plot(t_test, test_preds, marker="o", label="Test Prediction")
#    plt.plot(t_future, future, marker="o", label="Future Forecast")

    # mark split point on the original timeline (approximate visual)
#    plt.axvline(x=t_test[0] - 1, linestyle="--")

#    plt.xlabel("Time (t)")
#    plt.ylabel("Value")
#    plt.title("Pure Python Random Forest Forecasting (Observed + Test + Future)")
#    plt.legend()
#    plt.tight_layout()
#    plt.show()
#    import random
#    import matplotlib.pyplot as plt


    # =========================================================
    # Utilities
    # =========================================================
    def mean(values):
        return sum(values) / len(values) if values else 0.0


    def variance(values):
        if not values:
            return 0.0
        m = mean(values)
        return sum((v - m) ** 2 for v in values) / len(values)


    def mse_of_split(left_y, right_y):
        n = len(left_y) + len(right_y)
        if n == 0:
            return float("inf")
        return (len(left_y) / n) * variance(left_y) + (len(right_y) / n) * variance(right_y)


    # =========================================================
    # CART-like Regression Tree (pure python)
    # =========================================================
    class TreeNode:
        __slots__ = ("feature", "threshold", "left", "right", "value")

        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

        def is_leaf(self):
            return self.value is not None


    class DecisionTreeRegressorPure:
        def __init__(self, max_depth=6, min_samples_split=8, max_features=None, random_state=None):
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.max_features = max_features
            self.root = None
            self._rng = random.Random(random_state)

        def fit(self, X, y):
            self.root = self._build_tree(X, y, depth=0)
            return self

        def predict_one(self, x):
            node = self.root
            while not node.is_leaf():
                if x[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            return node.value

        def _best_split(self, X, y):
            n_features = len(X[0])

            # random subset of features
            feat_count = n_features if self.max_features is None else min(self.max_features, n_features)
            features = list(range(n_features))
            self._rng.shuffle(features)
            features = features[:feat_count]

            best_feature = None
            best_threshold = None
            best_score = float("inf")

            for f in features:
                vals = sorted(set(row[f] for row in X))
                if len(vals) <= 1:
                    continue

                thresholds = [(vals[i] + vals[i + 1]) / 2.0 for i in range(len(vals) - 1)]
                for thr in thresholds:
                    left_y, right_y = [], []
                    for row, target in zip(X, y):
                        (left_y if row[f] <= thr else right_y).append(target)

                    if not left_y or not right_y:
                        continue

                    score = mse_of_split(left_y, right_y)
                    if score < best_score:
                        best_score = score
                        best_feature = f
                        best_threshold = thr

            return best_feature, best_threshold

        def _build_tree(self, X, y, depth):
            if (depth >= self.max_depth) or (len(y) < self.min_samples_split) or (variance(y) < 1e-9):
                return TreeNode(value=mean(y))

            feature, threshold = self._best_split(X, y)
            if feature is None:
                return TreeNode(value=mean(y))

            left_X, left_y, right_X, right_y = [], [], [], []
            for row, target in zip(X, y):
                if row[feature] <= threshold:
                    left_X.append(row);
                    left_y.append(target)
                else:
                    right_X.append(row);
                    right_y.append(target)

            if not left_y or not right_y:
                return TreeNode(value=mean(y))

            left_node = self._build_tree(left_X, left_y, depth + 1)
            right_node = self._build_tree(right_X, right_y, depth + 1)
            return TreeNode(feature=feature, threshold=threshold, left=left_node, right=right_node)


    # =========================================================
    # Random Forest Regressor (bagging + random features)
    # =========================================================
    class RandomForestRegressorPure:
        def __init__(self, n_estimators=80, max_depth=6, min_samples_split=6,
                     max_features=3, random_state=42):
            self.n_estimators = n_estimators
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.max_features = max_features
            self._rng = random.Random(random_state)
            self.trees = []

        def _bootstrap(self, X, y):
            n = len(y)
            Xb, yb = [], []
            for _ in range(n):
                i = self._rng.randrange(n)
                Xb.append(X[i])
                yb.append(y[i])
            return Xb, yb

        def fit(self, X, y):
            self.trees = []
            for _ in range(self.n_estimators):
                Xb, yb = self._bootstrap(X, y)
                tree = DecisionTreeRegressorPure(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    max_features=self.max_features,
                    random_state=self._rng.randrange(10 ** 9)
                )
                tree.fit(Xb, yb)
                self.trees.append(tree)
            return self

        def predict_one(self, x):
            return mean([t.predict_one(x) for t in self.trees])

        def predict(self, X):
            return [self.predict_one(x) for x in X]


    # =========================================================
    # Time series -> supervised (lag features) + recursive forecast
    # =========================================================
    def make_lags(series, lags):
        X, Y = [], []
        for t in range(lags, len(series)):
            X.append([series[t - i] for i in range(1, lags + 1)])
            Y.append(series[t])
        return X, Y


    def recursive_forecast(model, history, lags, h):
        hist = list(history)
        out = []
        for _ in range(h):
            x = [hist[-i] for i in range(1, lags + 1)]
            yhat = model.predict_one(x)
            out.append(yhat)
            hist.append(yhat)
        return out


    # =========================================================
    # Demo on your series + plotting
    # =========================================================
    if __name__ == "__main__":
        series = [
            280.05, 305.18, 324.14, 355.64, 394.53, 438.96, 501.13, 572.49, 635.81,
            726.68, 784.94, 727.34, 713.08, 747.34, 771.44, 894.97, 894.46, 968.30,
            1012.44, 1046.07, 1132.78, 1252.69, 1187.70, 1205.45, 1241.29, 1319.69,
            1426.17, 1856.83, 2023.56, 2129.56, 2239.85, 2389.03, 2660.00, 2770.00,
            2863.00, 3228.00, 3557.00, 3894.00, 4171.00, 4580.00, 4644.00, 4621.00,
            5142.00, 6440.00, 6448.00, 5960.00, 6155.00, 6507.00, 6001.00, 5927.00,
            6731.00, 7614.00, 7103.00, 6746.00, 7328.00, 8212.00, 7993.00, 8300.00,
            8731.00, 8702.00, 6677.00
        ]

        lags = 6
        test_size = 8
        horizon = 8

        X, y = make_lags(series, lags)

        # split (note: X/y start at t=lags+1)
        X_train, y_train = X[:-test_size], y[:-test_size]
        X_test, y_test = X[-test_size:], y[-test_size:]

        rf = RandomForestRegressorPure(
            n_estimators=80,
            max_depth=6,
            min_samples_split=6,
            max_features=3,
            random_state=42
        )
        rf.fit(X_train, y_train)

        test_preds = rf.predict(X_test)
        future = recursive_forecast(rf, series, lags, horizon)

        # --- Build time axes for plotting ---
        n = len(series)
        t_obs = list(range(1, n + 1))

        # test predictions correspond to last "test_size" targets in y
        # y indices align with original series at t = lags+1 ... n
        # so the last test_size predictions correspond to times:
        start_test_t = (lags + 1) + (len(y) - test_size)
        t_test = list(range(start_test_t, start_test_t + test_size))

        t_future = list(range(n + 1, n + horizon + 1))
        print("Future:",future)
        # --- Plot ---
        plt.figure(figsize=(11, 5))
        plt.plot(t_obs, series, label="Observed")
        plt.plot(t_test, test_preds, marker="o", label="Test Prediction")
        plt.plot(t_future, future, marker="o", label="Future Forecast")

        # mark split point on the original timeline (approximate visual)
        plt.axvline(x=t_test[0] - 1, linestyle="--")

        plt.xlabel("Time (t)")
        plt.ylabel("Value")
        plt.title("Pure Python Random Forest Forecasting (Observed + Test + Future)")
        plt.legend()
        plt.tight_layout()
        plt.show()
