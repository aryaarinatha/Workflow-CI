import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Muat data terproses dan encoder label
train_df = pd.read_csv("song_preprocessing/song_train_processed.csv")
test_df = pd.read_csv("song_preprocessing/song_test_processed.csv")
rating_le = joblib.load("song_preprocessing/rating_le.pkl")
label_names = rating_le.classes_

# Bagi fitur dan target
X_train = train_df.drop(columns=["Rating_class_encoded"])
y_train = train_df["Rating_class_encoded"]
X_test = test_df.drop(columns=["Rating_class_encoded"])
y_test = test_df["Rating_class_encoded"]

mlflow.sklearn.autolog()

with mlflow.start_run() as run:
    # Simpan run id untuk inference
    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)

    model = RandomForestClassifier(n_estimators=100, random_state=7)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions, target_names=label_names))