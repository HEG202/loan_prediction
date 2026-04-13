import pandas as pd
from db_conn import open_db, close_db
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class LoanDataManager:
    def __init__(self, file_name):
        self.file_name = file_name
        self.df = None
        self.conn = None
        self.cur = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None

    def load_csv(self):
        self.df = pd.read_csv(self.file_name)
        print("CSV 파일 읽기 완료")
        # print(self.df.head())

    def handle_missing_values(self):
        # 범주형: 최빈값
        self.df["Gender"] = self.df["Gender"].fillna(self.df["Gender"].mode()[0])
        self.df["Married"] = self.df["Married"].fillna(self.df["Married"].mode()[0])
        self.df["Dependents"] = self.df["Dependents"].fillna(self.df["Dependents"].mode()[0])
        self.df["Self_Employed"] = self.df["Self_Employed"].fillna(self.df["Self_Employed"].mode()[0])

        # 수치형: 중앙값 / 최빈값
        self.df["LoanAmount"] = self.df["LoanAmount"].fillna(self.df["LoanAmount"].median())
        self.df["Loan_Amount_Term"] = self.df["Loan_Amount_Term"].fillna(
            self.df["Loan_Amount_Term"].mode()[0]
        )

        # 별도 처리
        self.df["Credit_History"] = self.df["Credit_History"].fillna(-1)

        print("결측치 처리 완료")
        # print(self.df.isnull().sum())

    def connect_db(self):
        self.conn, self.cur = open_db()
        print("DB 연결 완료")

    def create_table(self):
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS loan_train (
            Loan_ID VARCHAR(20) PRIMARY KEY,
            Gender VARCHAR(10),
            Married VARCHAR(10),
            Dependents VARCHAR(10),
            Education VARCHAR(20),
            Self_Employed VARCHAR(10),
            ApplicantIncome INT,
            CoapplicantIncome FLOAT,
            LoanAmount FLOAT,
            Loan_Amount_Term FLOAT,
            Credit_History FLOAT,
            Property_Area VARCHAR(20),
            Loan_Status VARCHAR(1)
        )
        """
        self.cur.execute(create_table_sql)
        self.conn.commit()
        print("loan_train 테이블 생성 완료")

    def insert_data(self):
        self.cur.execute("DELETE FROM loan_train")
        self.conn.commit()

        insert_sql = """
        INSERT INTO loan_train (
            Loan_ID, Gender, Married, Dependents, Education, Self_Employed,
            ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
            Credit_History, Property_Area, Loan_Status
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        rows = [row for row in self.df.itertuples(index=False, name=None)]
        self.cur.executemany(insert_sql, rows)
        self.conn.commit()

        print(f"{len(rows)} rows inserted into loan_train")

    def check_row_count(self):
        self.cur.execute("SELECT COUNT(*) AS cnt FROM loan_train")
        result = self.cur.fetchone()
        print("현재 저장된 행 개수:", result["cnt"])

    def load_from_db(self):
        sql = "SELECT * FROM loan_train"
        self.cur.execute(sql)
        rows = self.cur.fetchall()

        self.df = pd.DataFrame(rows)

        print("DB에서 데이터 불러오기 완료")
        # print("데이터 크기:", self.df.shape)
        # print(self.df.head())

    def close(self):
        if self.conn is not None and self.cur is not None:
            close_db(self.conn, self.cur)
            print("DB 연결 종료")
    
    def plot_basic_distributions(self):
        numeric_columns = [
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Loan_Amount_Term",
            "Credit_History"
        ]

        for col in numeric_columns:
            plt.figure(figsize=(8, 5))
            plt.hist(self.df[col], bins=20, edgecolor="black")
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.3)
            plt.show()

    def plot_categorical_distributions(self):
        categorical_columns = [
            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Self_Employed",
            "Property_Area",
            "Loan_Status"
        ]

        for col in categorical_columns:
            plt.figure(figsize=(8, 5))
            sns.countplot(data=self.df, x=col)
            plt.title(f"Count of {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.xticks(rotation=20)
            plt.show()

    def plot_scatter_selected(self):
        scatter_pairs = [
            ("ApplicantIncome", "LoanAmount"),
            ("ApplicantIncome", "CoapplicantIncome"),
            ("CoapplicantIncome", "LoanAmount")
        ]

        colors = {"Y": "blue", "N": "red"}

        for x_col, y_col in scatter_pairs:
            plt.figure(figsize=(8, 6))

            for status, color in colors.items():
                subset = self.df[self.df["Loan_Status"] == status]
                plt.scatter(
                    subset[x_col],
                    subset[y_col],
                    label=f"Loan_Status {status}",
                    alpha=0.6,
                    color=color
                )

            plt.title(f"{x_col} vs {y_col}")
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

    def plot_correlation_heatmap(self):
        temp_df = self.df.copy()

        temp_df["Loan_Status"] = temp_df["Loan_Status"].map({"Y": 1, "N": 0})

        numeric_df = temp_df[
            [
                "ApplicantIncome",
                "CoapplicantIncome",
                "LoanAmount",
                "Loan_Amount_Term",
                "Credit_History",
                "Loan_Status"
            ]
        ]

        corr_matrix = numeric_df.corr()

        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.show()

    def plot_categorical_vs_loan_status(self):
        categorical_columns = [
            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Self_Employed",
            "Property_Area",
            "Credit_History"
        ]

        for col in categorical_columns:
            plt.figure(figsize=(8, 5))
            sns.countplot(data=self.df, x=col, hue="Loan_Status")
            plt.title(f"{col} by Loan_Status")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.xticks(rotation=20)
            plt.legend(title="Loan_Status")
            plt.show()

    def plot_numeric_boxplots_by_loan_status(self):
        numeric_columns = [
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Loan_Amount_Term"
        ]

        for col in numeric_columns:
            plt.figure(figsize=(8, 5))
            sns.boxplot(data=self.df, x="Loan_Status", y=col)
            plt.title(f"{col} by Loan_Status")
            plt.xlabel("Loan_Status")
            plt.ylabel(col)
            plt.show()

    def preprocess_for_model(self):
        model_df = self.df.copy()

        # 1. target 변환
        model_df["Loan_Status"] = model_df["Loan_Status"].map({"Y": 1, "N": 0})

        # 2. Dependents 변환
        model_df["Dependents"] = model_df["Dependents"].replace("3+", 3).astype(int)

        # 3. 범주형 변수 one-hot encoding
        categorical_columns = [
            "Gender",
            "Married",
            "Education",
            "Self_Employed",
            "Property_Area"
        ]

        model_df = pd.get_dummies(model_df, columns=categorical_columns, drop_first=True)

        # 4. X, y 분리
        self.X = model_df.drop(["Loan_ID", "Loan_Status"], axis=1)
        self.y = model_df["Loan_Status"]

        # 5. bool 타입 dummy 컬럼을 0/1로 변환
        bool_cols = self.X.select_dtypes(include=["bool"]).columns
        self.X[bool_cols] = self.X[bool_cols].astype(int)

        print("모델링용 전처리 완료")
        print("X shape:", self.X.shape)
        print("y shape:", self.y.shape)
        # print(self.X.head())
        # print(self.y.head())

    def split_train_test(self, test_size=0.3, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=random_state
        )

        print("train/test split 완료")
        print("X_train shape:", self.X_train.shape)
        print("X_test shape:", self.X_test.shape)
        print("y_train shape:", self.y_train.shape)
        print("y_test shape:", self.y_test.shape)

    def evaluate_classification_performance(self):
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred)

        print("분류 성능 평가")
        print(f"accuracy = {accuracy:.4f}")
        print(f"precision = {precision:.4f}")
        print(f"recall = {recall:.4f}")
        print(f"f1 score = {f1:.4f}")

    def train_and_test_lr_model(self):
        lr = LogisticRegression(max_iter=5000)
        lr.fit(self.X_train, self.y_train)

        self.y_pred = lr.predict(self.X_test)

        print("Logistic Regression 학습 및 예측 완료")

    def train_and_test_knn_model(self):
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(self.X_train, self.y_train)

        self.y_pred = knn.predict(self.X_test)

        print("KNN 학습 및 예측 완료")

    def train_and_test_svm_model(self):
        svm = SVC()
        svm.fit(self.X_train, self.y_train)

        self.y_pred = svm.predict(self.X_test)

        print("SVM 학습 및 예측 완료")

    def lr_kfold_performance(self):
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        lr = LogisticRegression(max_iter=5000)

        cv_results = cross_validate(
            lr,
            self.X,
            self.y,
            cv=kfold,
            scoring=["accuracy", "precision", "recall", "f1"]
        )

        print("\n[Logistic Regression - KFold]")
        for metric, scores in cv_results.items():
            if metric.startswith("test_"):
                print(f"{metric[5:]}: {scores.mean():.4f}")

    def knn_kfold_performance(self):
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        knn = KNeighborsClassifier(n_neighbors=5)

        cv_results = cross_validate(
            knn,
            self.X,
            self.y,
            cv=kfold,
            scoring=["accuracy", "precision", "recall", "f1"]
        )

        print("\n[KNN - KFold]")
        for metric, scores in cv_results.items():
            if metric.startswith("test_"):
                print(f"{metric[5:]}: {scores.mean():.4f}")

    def svm_kfold_performance(self):
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        svm = SVC()

        cv_results = cross_validate(
            svm,
            self.X,
            self.y,
            cv=kfold,
            scoring=["accuracy", "precision", "recall", "f1"]
        )

        print("\n[SVM - KFold]")
        for metric, scores in cv_results.items():
            if metric.startswith("test_"):
                print(f"{metric[5:]}: {scores.mean():.4f}")

    def apply_standard_scaling(self):
        scaler = StandardScaler()

        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)

        print("Standard Scaling 완료")
        print("X_train_scaled shape:", self.X_train_scaled.shape)
        print("X_test_scaled shape:", self.X_test_scaled.shape)

    def train_and_test_lr_model_scaled(self):
        lr = LogisticRegression(max_iter=5000)
        lr.fit(self.X_train_scaled, self.y_train)

        self.y_pred = lr.predict(self.X_test_scaled)

        print("Scaled Logistic Regression 학습 및 예측 완료")

    def train_and_test_knn_model_scaled(self):
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(self.X_train_scaled, self.y_train)

        self.y_pred = knn.predict(self.X_test_scaled)

        print("Scaled KNN 학습 및 예측 완료")

    def train_and_test_svm_model_scaled(self):
        svm = SVC()
        svm.fit(self.X_train_scaled, self.y_train)

        self.y_pred = svm.predict(self.X_test_scaled)

        print("Scaled SVM 학습 및 예측 완료")

    def plot_confusion_matrix(self, title="Confusion Matrix"):
        cm = confusion_matrix(self.y_test, self.y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.title(title)
        plt.show()

if __name__ == "__main__":
    manager = LoanDataManager("train_csv.csv")
    manager.load_csv()
    manager.handle_missing_values()
    manager.connect_db()
    manager.create_table()
    manager.insert_data()
    manager.check_row_count()
    manager.load_from_db()

    # manager.plot_basic_distributions()
    # manager.plot_categorical_distributions()
    # manager.plot_categorical_vs_loan_status()
    # manager.plot_numeric_boxplots_by_loan_status()
    # manager.plot_scatter_selected()
    # manager.plot_correlation_heatmap()

    manager.preprocess_for_model()
    manager.split_train_test()

    # manager.train_and_test_lr_model()
    # manager.evaluate_classification_performance()

    # manager.train_and_test_knn_model()
    # manager.evaluate_classification_performance()

    manager.train_and_test_svm_model()
    manager.evaluate_classification_performance()
    manager.plot_confusion_matrix(title="SVM Confusion Matrix (Before Scaling)")

    # manager.lr_kfold_performance()
    # manager.knn_kfold_performance()
    # manager.svm_kfold_performance()

    manager.apply_standard_scaling()

    # manager.train_and_test_lr_model_scaled()
    # manager.evaluate_classification_performance()

    # manager.train_and_test_knn_model_scaled()
    # manager.evaluate_classification_performance()

    manager.train_and_test_svm_model_scaled()
    manager.evaluate_classification_performance()
    manager.plot_confusion_matrix(title="SVM Confusion Matrix (Before Scaling)")

    manager.close()
