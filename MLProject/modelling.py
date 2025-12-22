import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df_train = pd.read_csv("song_preprocessing/song_train_processed.csv")
df_test = pd.read_csv("song_preprocessing/song_test_processed.csv")
rating_le = joblib.load("song_preprocessing/rating_le.pkl")
class_names = rating_le.classes_

X_train = df_train.drop(columns=["Rating_class_encoded"])
y_train = df_train["Rating_class_encoded"]
X_test = df_test.drop(columns=["Rating_class_encoded"])
y_test = df_test["Rating_class_encoded"]

mlflow.set_tracking_uri("")
mlflow.set_experiment("Song_Rating_Classification")
mlflow.sklearn.autolog()

with mlflow.start_run() as run:
    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=7
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=class_names))