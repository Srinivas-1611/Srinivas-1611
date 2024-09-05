import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sqlalchemy import create_engine


# Database connection using SQLAlchemy
db_connection_string = "mysql+mysqlconnector://root:srini123@localhost/fmdc_db"
engine = create_engine(db_connection_string)


# Load data from the database
def load_data():
    query = "SELECT * FROM fmdc_data"
    df = pd.read_sql(query, engine)
    return df


# Data Preprocessing and Model Training
def preprocess_and_train(df):
    # Handle missing values in the target column
    df['risk_class'] = df['risk_class'].fillna(df['risk_class'].mode()[0])

    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Label encoding for categorical features
    encoders = {col: LabelEncoder() for col in categorical_cols}
    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')
        encoders[col].fit(df[col].unique())
        df[col] = encoders[col].transform(df[col])

    # Handle missing values using the most frequent value
    imputer = SimpleImputer(strategy='most_frequent')

    X = df.drop('risk_class', axis=1)
    y = df['risk_class']

    imputer.fit(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print(classification_report(y_test, y_pred))

    conf_matrix = confusion_matrix(y_test, y_pred)

    return rf_classifier, encoders, imputer, X.columns


# Load initial data and train the model
df = load_data()
rf_classifier, encoders, imputer, feature_columns = preprocess_and_train(df)

# Dictionaries with risk class information
risk_class_descriptions = {
    1: "A situation where there is a reasonable chance that a product will cause serious health problems or death.",
    2: "A situation where a product may cause a temporary or reversible health problem or where there is a slight chance that it will cause serious health problems or death.",
    3: "A situation where a product is not likely to cause any health problem or injury."
}

risk_class_suggestions = {
    1: "Immediate action is required to address the issue. Consider recalling the product or performing urgent maintenance to prevent serious outcomes.",
    2: "Monitor the situation closely and schedule maintenance to address potential issues. A temporary or reversible health problem may occur, but serious risks are low.",
    3: "Routine maintenance is sufficient. The product is not likely to cause any health problems or injury, so immediate action is not necessary."
}


# Prediction function
def predict_new_data(new_data, confidence_threshold=0.6):
    try:
        current_categorical_cols = [col for col in encoders.keys() if col in new_data.columns]

        for col in current_categorical_cols:
            new_data[col] = new_data[col].fillna('Unknown')
            unseen_labels = set(new_data[col].unique()) - set(encoders[col].classes_)
            if unseen_labels:
                new_labels = list(encoders[col].classes_) + list(unseen_labels)
                encoders[col].classes_ = np.array(new_labels)
            new_data[col] = encoders[col].transform(new_data[col])

        new_data = pd.DataFrame(imputer.transform(new_data), columns=feature_columns)

        probabilities = rf_classifier.predict_proba(new_data)[0]
        predicted_class = rf_classifier.classes_[np.argmax(probabilities)]
        max_probability = np.max(probabilities)

        description = risk_class_descriptions[predicted_class]
        suggestion = risk_class_suggestions[predicted_class]

        return predicted_class, description, suggestion

    except Exception as e:
        return f"Error in prediction: {str(e)}", "", ""


# Retrain model function after new data insertion
def retrain_model():
    global rf_classifier, encoders, imputer, feature_columns
    df = load_data()
    rf_classifier, encoders, imputer, feature_columns = preprocess_and_train(df)

