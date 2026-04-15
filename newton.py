import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ====================== SIGMOID ======================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ====================== COST ======================
def compute_cost(X, y, theta, lambda_=1e-4):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5

    cost = (-1/m) * np.sum(y*np.log(h+epsilon) + (1-y)*np.log(1-h+epsilon))
    reg = (lambda_/(2*m)) * np.sum(theta[1:]**2)
    return cost + reg

# ====================== NEWTON OPTIMIZED ======================
def newton_optimized(X, y, theta, iterations=12, lambda_=1e-4):
    m, n = X.shape
    cost_history = []

    print("Training bằng Newton Method (Optimized)...")

    for i in range(iterations):
        h = sigmoid(X @ theta)

        # Gradient
        grad = (1/m) * (X.T @ (h - y))
        grad[1:] += (lambda_/m) * theta[1:]

        # R vector (không tạo diag)
        R = h * (1 - h)

        # Hessian tối ưu
        H = (1/m) * (X.T @ (X * R[:, np.newaxis]))

        # Regularization
        H += lambda_ * np.eye(n)

        # Solve
        try:
            delta = np.linalg.solve(H, grad)
            theta = theta - delta
        except np.linalg.LinAlgError:
            print("Hessian lỗi, dừng sớm")
            break

        cost = compute_cost(X, y, theta, lambda_)
        cost_history.append(cost)

        print(f"Iteration {i+1}/{iterations} - Cost = {cost:.6f}")

    return theta, cost_history


# ====================== MAIN ======================
print("=== SPAM DETECTION - FILE LOCAL ===")

#  DÙNG FILE
df = pd.read_csv('spam_ham_dataset.csv', encoding='latin-1')

print("Columns:", df.columns.tolist())
print("Shape:", df.shape)

# ====================== XỬ LÝ DATA LINH HOẠT ======================

# Tìm cột text
if 'text' in df.columns:
    text_col = 'text'
elif 'message' in df.columns:
    text_col = 'message'
elif 'sms' in df.columns:
    text_col = 'sms'
else:
    text_col = df.columns[-1]  # fallback
    print(f"Không tìm thấy cột text chuẩn → dùng: {text_col}")

# Tìm label
if 'label_num' in df.columns:
    y = df['label_num'].values.astype(float)
elif 'label' in df.columns:
    df['label'] = df['label'].astype(str).str.lower().str.strip()
    y = df['label'].map({'ham': 0, 'spam': 1}).values.astype(float)
else:
    raise Exception("Không tìm thấy cột label")

X_text = df[text_col].astype(str)

print("Dùng cột text:", text_col)

# ====================== TF-IDF ======================
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=1500,   # giảm để chạy mượt
    min_df=2,
    max_df=0.95
)

X_vec = vectorizer.fit_transform(X_text).toarray()

# ====================== SPLIT ======================
np.random.seed(42)
idx = np.random.permutation(len(df))
X_vec = X_vec[idx]
y = y[idx]

split = int(0.8 * len(df))

X_train = X_vec[:split]
X_test  = X_vec[split:]
y_train = y[:split]
y_test  = y[split:]

# ====================== ADD BIAS ======================
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test  = np.c_[np.ones(X_test.shape[0]), X_test]

# ====================== TRAIN ======================
theta = np.zeros(X_train.shape[1])

theta_final, cost_history = newton_optimized(
    X_train, y_train, theta,
    iterations=10,
    lambda_=1e-4
)

# ====================== PLOT ======================
plt.plot(cost_history)
plt.title("Cost (Newton Optimized)")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.grid()
plt.show()

# ====================== PREDICT ======================
h_test = sigmoid(X_test @ theta_final)
y_pred = (h_test >= 0.5).astype(int)

# ====================== EVALUATE ======================
print("\n=== REPORT ===")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
