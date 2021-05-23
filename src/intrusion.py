import pandas as pd
import pathlib

from joblib import dump, load
from sklearn import metrics

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans, MiniBatchKMeans, Birch


def print_scores(y_true, y_pred):
    print("\tAccuracy: {:.5f}".format(metrics.accuracy_score(y_true, y_pred)))
    print("\tPrecision: {:.5f}".format(metrics.precision_score(y_true, y_pred)))
    print("\tRecall: {:.5f}".format(metrics.recall_score(y_true, y_pred)))
    print("\tF1 score: {:.5f}".format(metrics.f1_score(y_true, y_pred)))


def fit_predict(x_train, x_test, model, file_to_save):
    # Check if exists and load the fitted classifier else fit him and save it for later use
    fitted_clf_path = pathlib.Path(file_to_save)
    if fitted_clf_path.exists():
        print("Loading the fitted model for classifier: {}".format(model))
        classifier = load(file_to_save)
    else:
        classifier = model
        classifier.fit(x_train)
        print("Fitted classifier {} and saving it".format(model))
        # Save classifier
        dump(classifier, file_to_save)

    y_predicted = classifier.predict(x_test)
    return y_predicted


def encode_labels(labels: pd.DataFrame, encoding: dict):
    encoded_labels = labels.replace(encoding)
    return encoded_labels.to_numpy()


# Read data
X_train = pd.read_csv("NSL-KDDTrain.csv")
test_df = pd.read_csv("NSL-KDDTest.csv")

# Drop any examples that may contain NaN features
X_train.dropna(inplace=True)
test_df.dropna(inplace=True)

y_true = test_df['target']
X_test = test_df.drop('target', axis=1)

# One Hot encoding for categorical features
categorical_features = ['protocol_type', 'service', 'flag']

# Separate numerical and categorical features to be encoded
X_train_cat = X_train[categorical_features]
X_train_numeric = X_train.drop(categorical_features, axis=1)

X_test_cat = X_test[categorical_features]
X_test_numeric = X_test.drop(categorical_features, axis=1)

# One-hot-encoding for categorical features
encoder = OneHotEncoder(handle_unknown='ignore')
train_encoded_features = encoder.fit_transform(X_train_cat).toarray()
test_encoded_features = encoder.transform(X_test_cat).toarray()

df_train_encoded_features = pd.DataFrame(data=train_encoded_features)
df_test_encoded_features = pd.DataFrame(data=test_encoded_features)

# Join back the numeric and categorical features per index
X_train_encoded = pd.concat([X_train_numeric, df_train_encoded_features], axis=1)
X_test_encoded = pd.concat([X_test_numeric, df_test_encoded_features], axis=1)

# Scale the data to default [0-1]. This technique leaves unmodified the one-hot-encoded features
scaler = MinMaxScaler()
X_train_encoded_scaled = scaler.fit_transform(X_train_encoded)
X_test_encoded_scaled = scaler.transform(X_test_encoded)


# Since we don't have labels we use unsupervised models.
# We try 2 techniques:
#   1. Outlier detection
#   2. Clustering

# Outlier detection
isolation_forest = IsolationForest(n_estimators=100, verbose=1, n_jobs=-1)
local_outlier_factor = LocalOutlierFactor(n_neighbors=20, n_jobs=-1)

y_isolation_forest_predicted = fit_predict(X_train_encoded_scaled, X_test_encoded_scaled, isolation_forest, "isolation-forest.joblib")
y_local_outlier_predicted = fit_predict(X_train_encoded_scaled, X_test_encoded_scaled, local_outlier_factor, "local-outlier-factor-clf.joblib")

# Encode inlier/outlier values
# 1 : normal (inlier), -1 : attack (outlier)
y_true_outlier_encoded = encode_labels(y_true, {'attack': -1, 'normal': 1})

print("For IsolationForest model")
print_scores(y_true_outlier_encoded, y_isolation_forest_predicted)

print("For LocalOutlierFactor model")
print_scores(y_true_outlier_encoded, y_isolation_forest_predicted)


# Clustering
k_means = KMeans(n_clusters=2, random_state=0)
mini_batch_k_means = MiniBatchKMeans(n_clusters=2, random_state=0)
birch = Birch(n_clusters=2)

y_k_means_predicted = fit_predict(X_train_encoded_scaled, X_test_encoded_scaled, k_means, "k-means.joblib")
y_mini_batch_k_means_predicted = fit_predict(X_train_encoded_scaled, X_test_encoded_scaled, mini_batch_k_means, "minibatc-k-means.joblib")
y_birch_predicted = fit_predict(X_train_encoded_scaled, X_test_encoded_scaled, birch, "birch.joblib")

# Encode target clusters
# cluster 1 : attack, 0 : normal
y_true_clustering_encoded = encode_labels(y_true, {'attack': 1, 'normal': 0})

print("For KMeans model")
print_scores(y_true_clustering_encoded, y_k_means_predicted)

print("For MiniBatchKMeans model")
print_scores(y_true_clustering_encoded, y_mini_batch_k_means_predicted)

print("For Birch model")
print_scores(y_true_clustering_encoded, y_birch_predicted)
