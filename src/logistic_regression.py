import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.01, epoch=100, batch_size=32, threshold=0.5, random_state=None):
        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size
        self.threshold = threshold
        self.random_state = random_state

        self.classes_ = None
        self.weights = None
        self.bias = None
        self.loss_history = []

    @property
    def _is_binary(self):
        return len(self.classes_) == 2

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        num_classes = len(self.classes_)

        if num_classes < 2:
            raise ValueError("y must contain at least 2 classes.")

        # 1. Unified 2D target matrix: (N, 1) for binary, (N, K) one-hot for multi-class
        if self._is_binary:
            num_cols = 1
            y_target = y_idx.astype(np.float64).reshape(-1, 1)
        else:
            num_cols = num_classes
            y_target = np.eye(num_classes, dtype=np.float64)[y_idx]

        # 2. Initialize parameters with consistent 2D shapes
        num_data, num_features = X.shape
        self.weights = np.zeros((num_features, num_cols))
        self.bias = np.zeros((1, num_cols))
        self.loss_history = []

        # 3. Single unified training loop (works for both binary and multi-class)
        rng = np.random.default_rng(self.random_state)
        num_batches = int(np.ceil(num_data / self.batch_size))

        for _ in range(self.epoch):
            indices = rng.permutation(num_data)

            for i in range(num_batches):
                batch_idx = indices[i * self.batch_size : (i + 1) * self.batch_size]
                batch_X = X[batch_idx]
                batch_y = y_target[batch_idx]

                batch_error = self._forward(batch_X) - batch_y
                self.weights -= self.lr * (batch_X.T @ batch_error) / len(batch_X)
                self.bias -= self.lr * np.mean(batch_error, axis=0, keepdims=True)

            self.loss_history.append(self.compute_loss(y_target, self._forward(X)))

        return self

    def _forward(self, X):
        logits = X @ self.weights + self.bias
        return self.sigmoid(logits) if self._is_binary else self.softmax(logits)

    def predict_proba(self, X):
        if self.weights is None:
            raise RuntimeError("Model must be fitted before calling predict_proba.")
        probs = self._forward(np.asarray(X, dtype=np.float64))
        return probs.ravel() if self._is_binary else probs

    def predict(self, X):
        probs = self.predict_proba(X)
        if self._is_binary:
            indices = (probs >= self.threshold).astype(int)
        else:
            indices = np.argmax(probs, axis=1)
        return self.classes_[indices]

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def softmax(self,batch_score):
        shifted = batch_score - np.max(batch_score,axis=1,keepdims=True)
        exp_score = np.exp(shifted)
        sum_exp = np.sum(exp_score,axis=1,keepdims=True)
        return exp_score/sum_exp

    def compute_loss(self, y, y_hat):
        y = np.asarray(y, dtype=np.float64)
        y_hat = np.asarray(y_hat, dtype=np.float64)
        eps = 1e-15
        y_hat = np.clip(y_hat, eps, 1 - eps)

        if y.ndim == 1 or y.shape[1] == 1:
            return float(-np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))
        return float(-np.sum(y * np.log(y_hat)) / len(y))