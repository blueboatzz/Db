import streamlit as st 
import os 
import joblib 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib 
import seaborn as sns
matplotlib.use('Agg')
import seaborn as sns 
import missingno as msno
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import joblib

@st.cache_data
def load_data(dataset):
    df = pd.read_csv(dataset)
    return df

Pregnancies = {'0-5':0,'5-10':1,'10-20':2}
Glucose = {'0-50':0,'50-100':1,'100-150':2,'150-200':3}
BloodPressure = {'0-40':0,'40-80':1,'80-130':2}
SkinThickness = {'0-30':0,'30-60':1,'60-100':2}
Insulin = {'0-250':0,'250-500':1,'500-900':2}
BMI = {'0-30':0,'30-70':1}
DiabetesPedigreeFunction = {'0.08-1.00':0,'1.00-3.00':1}
Age = {'21-40':0,'40-60':1,'60-90':2}
Outcome = {'0':0,'1':1}

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val==key:
            return value
def get_key(val,my_dict):
    for key,value in my_dict.items():
        if val==value:
            return key

def load_prediction_model(model_file):
    loaded_model = joblib.load(open(os.path.join( model_file),"rb"))
    return loaded_model
    

def median_target(data,var):
        temp = data[data[var].notnull()]
        temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
        return temp

def detect_outliers_lof(data, n_neighbors=10):
    # Create the LocalOutlierFactor model
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    
    # Fit the model and predict outliers
    outliers = lof.fit_predict(data)
    
    # Get the indices of outliers
    outlier_indices = data.index[outliers == -1]
    
    return outlier_indices

def set_insuline(row):
    if 16 <= row["Insulin"] <= 166:
        return "Normal"
    else:
        return "Abnormal"
model_file = "models/logistic_regression_model.pkl"
loaded_model = load_prediction_model(model_file)

def main():
    
    st.title("Diabetes Prediction And Prescription Generator")
    st.subheader("Built With Streamlit And Predicted Using Ml models")
    
    menu = ["EDA","DATA PREPROCESSING","FITTING","PREDICTION","ABOUT"]
    choices = st.sidebar.selectbox("SELECT ACTIVITIES",menu)
    
    
    if choices == 'EDA':     
        st.subheader("EDA")
        
        data = load_data('data/diabetes.csv')
        st.dataframe(data.head(10))
        
        if st.checkbox("Show Summary"):
            st.write(data.describe())
        
        if st.checkbox("Show Shape"):
            
            st.write(data.shape)
            
       
    
        if st.checkbox("Value Count Plot"):
            st.write(data['Outcome'].value_counts().plot(kind = 'bar'))
            st.pyplot()
            
        if st.checkbox("Pie Chart"):
            st.write(data['Outcome'].value_counts().plot.pie(autopct ="%1.1f%%"))
            st.pyplot()
            
        if st.checkbox("Show Histogram and Labels"):
             # Create a histogram using Matplotlib
            st.subheader('Histogram of Age')
            fig, ax = plt.subplots()
            sns.histplot(data['Age'], bins=10, kde=False, ax=ax)

    # Set axis labels
            ax.set_xlabel('Age', fontsize=10)
            ax.set_ylabel('Count', fontsize=10) 
            
            st.pyplot(fig)
            st.write("MAX AGE:", data['Age'].max())
            st.write("MIN AGE:", data['Age'].min())       
        if st.checkbox("Show DataFrame Columns"):
            st.write(data.columns)
            
        if st.checkbox("Plot Distributions"):
            fig, ax = plt.subplots(4, 2, figsize=(20, 20))
            sns.distplot(data['Pregnancies'], bins=20, ax=ax[0, 0], color="red")
            sns.distplot(data['Glucose'], bins=20, ax=ax[0, 1], color="red")
            sns.distplot(data['BloodPressure'], bins=20, ax=ax[1, 0], color="red")
            sns.distplot(data['SkinThickness'], bins=20, ax=ax[1, 1], color="red")
            sns.distplot(data['Insulin'], bins=20, ax=ax[2, 0], color="red")
            sns.distplot(data['BMI'], bins=20, ax=ax[2, 1], color="red")
            sns.distplot(data['DiabetesPedigreeFunction'], bins=20, ax=ax[3, 0], color="red")
            sns.distplot(data['Age'], bins=20, ax=ax[3, 1], color="red")

    # Show plot
            st.pyplot(fig)       
            
            st.title('Grouped Aggregations')
        if st.checkbox("Grouped Aggregations"):
# Groupby Outcome and calculate mean pregnancies
            st.subheader("Mean Pregnancies by Outcome:")
            mean_pregnancies = data.groupby("Outcome").agg({'Pregnancies':'mean'})
            st.write(mean_pregnancies)

# Groupby Outcome and calculate max pregnancies
            st.subheader("Max Pregnancies by Outcome:")
            max_pregnancies = data.groupby("Outcome").agg({'Pregnancies':'max'})
            st.write(max_pregnancies)

# Groupby Outcome and calculate mean glucose
            st.subheader("Mean Glucose by Outcome:")
            mean_glucose = data.groupby("Outcome").agg({'Glucose':'mean'})
            st.write(mean_glucose) 
            
        if st.checkbox("Outcome Distribution"):
            data['Outcome'] = data['Outcome'].map({0: 'Healthy', 1: 'Diabetes'})

            

# Plotting
            fig, ax = plt.subplots(1, 2, figsize=(18, 8))

# Pie chart
            data['Outcome'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
            ax[0].set_title('Target')
            ax[0].set_ylabel('')

# Countplot
            sns.countplot(x='Outcome', data=data, ax=ax[1])
            ax[1].set_title('Outcome')

# Annotations to clarify numeric labels
            st.write("0 represents 'Healthy'")
            st.write("1 represents 'Diabetes'")

# Show plot
            st.pyplot(fig)  
            
        if st.checkbox("Show Correlation Matrix"):
            st.subheader("Correlation Matrix")

    # Map 'Outcome' column to numerical values if necessary
            if data['Outcome'].dtype == 'object':
                data['Outcome'] = data['Outcome'].map({'Healthy': 0, 'Diabetes': 1})
    
    # Calculate and display the correlation matrix
            st.write(data.corr())
 
            
        if st.checkbox("Show Correlation Matrix Heatmap"):
            st.subheader("Correlation Matrix Heatmap")
            fig, ax = plt.subplots(figsize=[20, 15])
            sns.heatmap(data.corr(), annot=True, fmt='.2f', ax=ax, cmap='magma')
            ax.set_title("Correlation Matrix", fontsize=20)
            st.pyplot(fig)  
            
        if st.checkbox("Columns"):
            st.subheader("DataFrame Columns")
            st.write(data.columns.tolist())   
            st.write("EXPLORATORY DATA ANALYSIS COMPLETED")
            
        if st.checkbox("Accuracy Reports"):
            models_accuracy = {
            'Logistic Regression': joblib.load('models/predictions.pkl'),
            'KNN': joblib.load('models/predictions1.pkl'),
            'SVM': joblib.load('models/svm_predictions.pkl'),
            'Decision Tree': joblib.load('models/decision_tree_predictions.pkl'),
            'Gradient Boosting': joblib.load('models/gradient_boosting_predictions.pkl'),
            'Random Forest': joblib.load('models/random_forest_predictions.pkl'),
            'XGBoost': joblib.load('models/xgboost_predictions.pkl'),
            'SVM': joblib.load('models/svm_predictions.pkl'),
            # Add more models as needed
        }
            model_names = []
            percentages_0 = []
            percentages_1 = []
           
     
        
            for model_name,predictions in models_accuracy.items():
                
                predictions_series = pd.Series(predictions)
               
                value_counts = predictions_series.value_counts()

        # Plot the distribution of predicted labels as a pie chart for the current model
                fig, ax = plt.subplots(figsize=(2, 2))
                predictions_series.value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
                ax.set_aspect('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                ax.set_title(f"Distribution of Predicted Labels - {model_name}")
                st.pyplot(fig)
                
               
                if 0 in value_counts and 1 in value_counts:
                   st.write(f"Percentage of 0 for {model_name} that is not diabetic: {value_counts[0] / len(predictions) * 100:.1f}%")
                   st.write(f"Percentage of 1 for {model_name} that is diabetic: {value_counts[1] / len(predictions) * 100:.1f}%")
                   
                if 0 in value_counts and 1 in value_counts:
                    percentage_0 = value_counts[0] / len(predictions) * 100
                    percentage_1 = value_counts[1] / len(predictions) * 100
                    model_names.append(model_name)
                    percentages_0.append(f"{percentage_0:.1f}%")
                    percentages_1.append(f"{percentage_1:.1f}%")

    # Create a DataFrame to store the percentages
            data = {
                'Model': model_names,
                'Percentage of 0': percentages_0,
                'Percentage of 1': percentages_1
            }      
            df = pd.DataFrame(data)
            df_sorted = df.sort_values(by='Percentage of 0', ascending=False)

    # Display the table
            st.write(df_sorted.iloc[[0]])

    # Find the model with the lowest percentage of 1
            best_model = df_sorted.iloc[0]['Model']

    # Display the best model according to the dataset
            st.write(f"The best model according to the dataset is: {best_model}")
        
            
    if choices == 'DATA PREPROCESSING':
        st.subheader("DATA PREPROCESSING")
        data = load_data('data/diabetes.csv')                 
               
        columns_to_replace = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                          'BMI', 'DiabetesPedigreeFunction', 'Age']
        data[columns_to_replace] = data[columns_to_replace].replace(0, np.NaN)
    
    # Display the updated DataFrame
        if st.checkbox("Updated DataFrame"):
            st.subheader("DataFrame with Zeros Replaced by NaN")
            st.write(data)
            st.subheader("Missing Values Count")
            st.write(data.isnull().sum())
            
        if st.checkbox("DataFrame Head"):
            st.subheader("First Few Rows of the DataFrame")
            st.write(data.head())
        if st.checkbox(" Missing Values Bar Plot"):
            st.subheader("Missing Values Bar Plot")
            fig, ax = plt.subplots(figsize=(8, 4))  # Adjust figsize for the size of the plot
            msno.bar(data, color="orange", fontsize=9, ax=ax, labels = True, figsize=(8, 4))  # Adjust fontsize, sparkline, and figsize
            ax.grid(True, axis='y', linestyle='solid', linewidth=2, color='grey')  # Add gridlines
            st.pyplot(fig) 
    
        
        if st.checkbox("Calculate Median by Outcome"):
            st.subheader("Median by Outcome")
            var_to_calculate_median = st.selectbox("Select variable", data.columns)
            median_data = median_target(data,var_to_calculate_median)
            
            st.write(median_data)
            
        columns = data.columns.drop("Outcome")  # Drop 'Outcome' column
        for i in columns:
            median_data = median_target(data, i)
            median_values = dict(zip(median_data['Outcome'], median_data[i]))  # Create a dictionary of median values
        
        # Fill missing values based on the 'Outcome' category
            data.loc[(data['Outcome'] == 0) & (data[i].isnull()), i] = median_values[0]
            data.loc[(data['Outcome'] == 1) & (data[i].isnull()), i] = median_values[1] 
            
        if st.checkbox("DataFrame Head After Filling Missing Values"):
            st.subheader("First Few Rows of the DataFrame After Filling Missing Values")
            st.write(data.head())
            
        if st.checkbox("Missing Values Count After Filling"):
            st.subheader("Missing Values Count After Filling")
            st.write(data.isnull().sum())
            
        if st.checkbox("Pair Plot"):
            st.subheader("Pair Plot with Outcome")
            p = sns.pairplot(data, hue="Outcome")
            st.pyplot(p)
            
            
        if st.checkbox("Outlier"):
            
            
            st.subheader("Outlier Detection")
            outlier_results = []

            for feature in data.columns:
                Q1 = data[feature].quantile(0.25)
                Q3 = data[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                if (data[data[feature] > upper].any(axis=None)):
                    outlier_results.append((feature, "yes"))
                else:
                    outlier_results.append((feature, "no"))

# Display outlier detection results in a table format
            st.table(outlier_results) 
            
        if st.checkbox("Insulin Box Plot"):
            st.subheader("Insulin Box Plot")
            plt.figure(figsize=(8, 7))
            sns.boxplot(x=data["Insulin"], color="red")
            plt.xlabel('Insulin', fontsize=10)
            plt.ylabel('Count', fontsize=10)
            st.pyplot(plt)
            
        if st.checkbox("Outlier Insulin Box Plot"):
            
            st.subheader("Outlier Insulin Box Plot")
            Q1 = data.Insulin.quantile(0.25)
            Q3 = data.Insulin.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            data.loc[data['Insulin'] > upper, "Insulin"] = upper
            plt.figure(figsize=(8, 7))
            sns.boxplot(x=data["Insulin"], color="red")
            plt.xlabel('Insulin', fontsize=10)
            plt.ylabel('Count', fontsize=10)
            st.pyplot(plt)
        
        if st.checkbox("LOF"):
            lof = LocalOutlierFactor(n_neighbors=10)
            predictions = lof.fit_predict(data)

# Display the outlier predictions
            st.write("Outlier Predictions:")
            st.write(predictions)
           
            lof_df = data.copy()
            lof_df['Outlier'] = predictions
    
    # Display the dataframe with predictions
            st.write("DataFrame with Outlier Labels:")
            st.dataframe(lof_df)                       
        if st.checkbox("CHECK HEAD"):
            st.write("Input DataFrame (head):")
            st.dataframe(data.head())
            
        if st.checkbox("Show Boxplot for Pregnancies"):
            st.write("Boxplot for Pregnancies:")
            plt.figure(figsize=(8, 7))
            sns.boxplot(x=data["Pregnancies"], color="red")
            st.pyplot(plt)    
        if st.checkbox("Show Outliers"):
    # Local Outlier Factor
            lof = LocalOutlierFactor(n_neighbors=10)
            df_scores = lof.fit_predict(data)

    # Get the outlier scores
            outlier_scores = lof.negative_outlier_factor_

    # Sort and display the first 20 outlier scores
            sorted_scores = np.sort(outlier_scores)[:20]
            st.write("First 20 Outlier Scores:")
            st.write(sorted_scores)

        if st.checkbox("THRESHOLD VALUE"):
    # Local Outlier Factor
            lof = LocalOutlierFactor(n_neighbors=10)
            df_scores = lof.fit_predict(data)

    # Get the outlier scores
            outlier_scores = lof.negative_outlier_factor_

    # Calculate threshold value
            threshold = np.sort(outlier_scores)[7]

    # Print threshold value
            st.write("Threshold Value:")
            st.write(threshold) 

        if st.checkbox("Show DataFrame with Detected Outliers (Threshold 1)"):
            outlier = df_scores > threshold
            df = data[outlier]
            st.subheader("DataFrame with Detected Outliers (Threshold 1)")
            st.write(df.head())   
            
        if st.checkbox("Add NewBMI Column"):
    # Define BMI categories
            bmi_categories = ["Underweight", "Normal", "Overweight", "Obesity 1", "Obesity 2", "Obesity 3"]
            NewBMI = pd.Series(bmi_categories, dtype="category")
            st.write("NewBMI Series:", NewBMI)
    
    # Assign NewBMI values based on BMI ranges
            data['NewBMI'] = pd.cut(data["BMI"], bins=[0, 18.5, 24.9, 29.9, 34.9, 39.9, float('inf')], labels=bmi_categories, right=False)
    
    # Assign NewInsulinScore column using a function 'set_insuline'
            data['NewInsulinScore'] = data.apply(set_insuline, axis=1)

            st.subheader("DataFrame with NewBMI and NewInsulinScore Columns")
            st.write(data.head())

    # One hot encoding
            df_encoded = pd.get_dummies(data, columns=["NewBMI", "NewInsulinScore"], drop_first=True)
            st.subheader("DataFrame after One-Hot Encoding")
            st.write(df_encoded.head())

        if st.checkbox("Add NewGlucose Column"):
            st.subheader("DataFrame with NewGlucose Column")
    
            NewGlucose = pd.Series(["Low", "Normal", "Overweight", "Secret", "High"], dtype="category")
            data["NewGlucose"] = np.nan
            data.loc[data["Glucose"] <= 70, "NewGlucose"] = NewGlucose[0]
            data.loc[(data["Glucose"] > 70) & (data["Glucose"] <= 99), "NewGlucose"] = NewGlucose[1]
            data.loc[(data["Glucose"] > 99) & (data["Glucose"] <= 126), "NewGlucose"] = NewGlucose[2]
            data.loc[data["Glucose"] > 126, "NewGlucose"] = NewGlucose[3]
    
            st.write(data.head())
        
        if st.checkbox("Perform One-Hot Encoding"):
            df_encoded = pd.get_dummies(data, columns=["NewBMI", "NewInsulinScore", "NewGlucose"], drop_first=True)
            st.subheader("DataFrame after One-Hot Encoding")
            st.write(df_encoded.head())
            st.write("Columns:", df_encoded.columns)
            categorical_df = df_encoded[['NewBMI_Normal', 'NewBMI_Overweight', 'NewBMI_Obesity 1', 'NewBMI_Obesity 2', 
                                 'NewBMI_Obesity 3', 'NewInsulinScore_Normal', 'NewGlucose_Normal', 
                                 'NewGlucose_Overweight', 'NewGlucose_Secret']]
            st.subheader("Categorical DataFrame")
            st.write(categorical_df.head()) 
            
        if st.checkbox("Prepare Features and Target"):
            y = df_encoded['Outcome']
            X = df_encoded.drop(['Outcome','NewBMI_Normal', 'NewBMI_Overweight', 'NewBMI_Obesity 1', 'NewBMI_Obesity 2', 
                                 'NewBMI_Obesity 3', 'NewInsulinScore_Normal', 'NewGlucose_Normal', 
                                 'NewGlucose_Overweight', 'NewGlucose_Secret'], axis=1)
            cols = X.columns
            index = X.index
            st.subheader("Features and Target")
            st.write("Features (X) head:")
            st.write(X.head())
            st.write("Columns:", cols)
            st.write("Index:", index )
            
            transformer = RobustScaler().fit(X)
            X = transformer.transform(X)
            X = pd.DataFrame(X, columns=cols, index=index)
            st.subheader("Scaled Features (X)")
            st.write(X.head())
            
        
        if st.checkbox("Concatenate Scaled Features and Categorical DataFrame"):
            X = pd.concat([X, categorical_df], axis=1)
            st.subheader("Final Feature Set (X)")
            st.write(X.head())
            st.write("Columns:", X.columns)
            st.write("Index:", X.index)
        if st.checkbox("Fitting for various Machine Learning algorithms"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Scaling the features using StandardScaler
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
    
            st.subheader("Scaled Features (X_train, X_test)")
            st.write("X_train head:")
            st.write(X_train[:5])
            st.write("X_test head:")
            st.write(X_test[:5])
            st.write("Shape of X_train:", X_train.shape)
            st.write("Shape of X_test:", X_test.shape)
        
            
    if choices == 'FITTING':
        st.subheader("FITTING")
        data = load_data('data/diabetes.csv')                 
        if st.checkbox("Add NewBMI Column"):
    # Define BMI categories
            bmi_categories = ["Underweight", "Normal", "Overweight", "Obesity 1", "Obesity 2", "Obesity 3"]
            NewBMI = pd.Series(bmi_categories, dtype="category")
            st.write("NewBMI Series:", NewBMI)
    
    # Assign NewBMI values based on BMI ranges
            data['NewBMI'] = pd.cut(data["BMI"], bins=[0, 18.5, 24.9, 29.9, 34.9, 39.9, float('inf')], labels=bmi_categories, right=False)
    
    # Assign NewInsulinScore column using a function 'set_insuline'
            data['NewInsulinScore'] = data.apply(set_insuline, axis=1)

            st.subheader("DataFrame with NewBMI and NewInsulinScore Columns")
            st.write(data.head())

    # One hot encoding
            df_encoded = pd.get_dummies(data, columns=["NewBMI", "NewInsulinScore"], drop_first=True)
            st.subheader("DataFrame after One-Hot Encoding")
            st.write(df_encoded.head())

        if st.checkbox("Add NewGlucose Column"):
            st.subheader("DataFrame with NewGlucose Column")
    
            NewGlucose = pd.Series(["Low", "Normal", "Overweight", "Secret", "High"], dtype="category")
            data["NewGlucose"] = np.nan
            data.loc[data["Glucose"] <= 70, "NewGlucose"] = NewGlucose[0]
            data.loc[(data["Glucose"] > 70) & (data["Glucose"] <= 99), "NewGlucose"] = NewGlucose[1]
            data.loc[(data["Glucose"] > 99) & (data["Glucose"] <= 126), "NewGlucose"] = NewGlucose[2]
            data.loc[data["Glucose"] > 126, "NewGlucose"] = NewGlucose[3]
    
            st.write(data.head())
        
        if st.checkbox("Perform One-Hot Encoding"):
            df_encoded = pd.get_dummies(data, columns=["NewBMI", "NewInsulinScore", "NewGlucose"], drop_first=True)
            st.subheader("DataFrame after One-Hot Encoding")
            st.write(df_encoded.head())
            st.write("Columns:", df_encoded.columns)
            categorical_df = df_encoded[['NewBMI_Normal', 'NewBMI_Overweight', 'NewBMI_Obesity 1', 'NewBMI_Obesity 2', 
                                 'NewBMI_Obesity 3', 'NewInsulinScore_Normal', 'NewGlucose_Normal', 
                                 'NewGlucose_Overweight', 'NewGlucose_Secret']]
            st.subheader("Categorical DataFrame")
            st.write(categorical_df.head()) 
            
        if st.checkbox("Prepare Features and Target"):
            y = df_encoded['Outcome']
            X = df_encoded.drop(['Outcome','NewBMI_Normal', 'NewBMI_Overweight', 'NewBMI_Obesity 1', 'NewBMI_Obesity 2', 
                                 'NewBMI_Obesity 3', 'NewInsulinScore_Normal', 'NewGlucose_Normal', 
                                 'NewGlucose_Overweight', 'NewGlucose_Secret'], axis=1)
            cols = X.columns
            index = X.index
            st.subheader("Features and Target")
            st.write("Features (X) head:")
            st.write(X.head())
            st.write("Columns:", cols)
            st.write("Index:", index )
            
            transformer = RobustScaler().fit(X)
            X = transformer.transform(X)
            X = pd.DataFrame(X, columns=cols, index=index)
            st.subheader("Scaled Features (X)")
            st.write(X.head())
            
        
        if st.checkbox("Concatenate Scaled Features and Categorical DataFrame"):
            X = pd.concat([X, categorical_df], axis=1)
            st.subheader("Final Feature Set (X)")
            st.write(X.head())
            st.write("Columns:", X.columns)
            st.write("Index:", X.index)
    
        if st.checkbox("Fitting for various Machine Learning algorithms"):
           X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        
        # Scaling the features using StandardScaler
           scaler = StandardScaler()
           X_train_scaled = scaler.fit_transform(X_train)
           X_test_scaled = scaler.transform(X_test)
        
           st.subheader("Scaled Features (X_train, X_test)")
           st.write("X_train head:")
           st.write(X_train_scaled[:5])
           st.write("X_test head:")
           st.write(X_test_scaled[:5])
           st.write("Shape of X_train:", X_train_scaled.shape)
           st.write("Shape of X_test:", X_test_scaled.shape)
        
        # Creating new DataFrame for scaled X_train and y_train
           X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
           y_train_df = pd.DataFrame(y_train).reset_index(drop=True)
        
        # Combining X_train_scaled and y_train_df into one DataFrame
           train_df = pd.concat([X_train_scaled_df, y_train_df], axis=1)
           train_df.columns = list(X.columns) + ['Outcome']
        
           if st.button("Logistic Regression"):
    # Initialize the Logistic Regression model
            log_reg = LogisticRegression()

    # Fit the model to the training data
            log_reg.fit(X_train, y_train)

    # Make predictions on the test data
            y_pred = log_reg.predict(X_test)

    # Calculate and display the accuracy score on the test data
            log_reg_acc = accuracy_score(y_test, y_pred)
            st.write("Accuracy Score:", log_reg_acc)

    # Display the confusion matrix
            st.write("Confusion Matrix:")
            st.write(confusion_matrix(y_test, y_pred))

    # Display the classification report
            classification_rep = classification_report(y_test, y_pred)
            st.text(classification_rep)
            model_data = {
        'model': log_reg,
        'accuracy_score': log_reg_acc,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }        
            joblib.dump(model_data, 'logistic_regression_model.pkl')
            st.write("Logistic Regression model and evaluation metrics saved successfully.")
        if st.button("K-Nearest Neighbors"):
    # Initialize the KNN classifier
            knn = KNeighborsClassifier()

    # Fit the model to the training data
            knn.fit(X_train, y_train)

    # Make predictions on the test data
            y_pred = knn.predict(X_test)

    # Calculate and display the accuracy score on the training data
            train_accuracy = accuracy_score(y_train, knn.predict(X_train))
            st.write("Training Accuracy:", train_accuracy)

    # Calculate and display the accuracy score on the test data
            knn_acc = accuracy_score(y_test, y_pred)
            st.write("Test Accuracy:", knn_acc)

    # Display the confusion matrix
            st.write("Confusion Matrix:")
            st.write(confusion_matrix(y_test, y_pred))

    # Display the classification report
            classification_rep = classification_report(y_test, y_pred)
            st.text(classification_rep)        
            model_data = {
        'model': knn,
        'training_accuracy': train_accuracy,
        'test_accuracy': knn_acc,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }

    # Save the model data in a PKL file
            joblib.dump(model_data, 'knn_model.pkl')
            st.write("K-Nearest Neighbors model and evaluation metrics saved successfully.")
        if st.button("Support Vector Machine"):
    # Define the SVM classifier with probability=True to enable probability estimates
            svc = SVC(probability=True)

    # Define the parameter grid for hyperparameter tuning
            parameter = {
        "gamma": [0.0001, 0.001, 0.01, 0.1],
        'C': [0.01, 0.05, 0.5, 0.01, 1, 10, 15, 20]
    }

    # Perform grid search using cross-validation
            grid_search = GridSearchCV(svc, parameter)
            grid_search.fit(X_train, y_train)

    # Print the best parameters and best score found during grid search
            st.write("Best Parameters:", grid_search.best_params_)
            st.write("Best Score:", grid_search.best_score_)

    # Initialize the SVM classifier with the best parameters found
            svc_best = SVC(C=grid_search.best_params_['C'], gamma=grid_search.best_params_['gamma'], probability=True)

    # Fit the SVM classifier to the training data
            svc_best.fit(X_train, y_train)

    # Make predictions on the test data
            y_pred = svc_best.predict(X_test)

    # Calculate and display the accuracy score on the training data
            train_accuracy = accuracy_score(y_train, svc_best.predict(X_train))
            st.write("Training Accuracy:", train_accuracy)

    # Calculate and display the accuracy score on the test data
            svc_acc = accuracy_score(y_test, y_pred)
            st.write("Test Accuracy:", svc_acc)

            # Display the confusion matrix
            st.write("Confusion Matrix:")
            st.write(confusion_matrix(y_test, y_pred))

    # Display the classification report
            classification_rep = classification_report(y_test, y_pred)
            st.text(classification_rep)
        if st.button("Decision Tree"):
    # Define the decision tree classifier
            DT = DecisionTreeClassifier()

    # Fit the decision tree classifier to the training data
            DT.fit(X_train, y_train)

    # Make predictions on the test data
            y_pred = DT.predict(X_test)

    # Calculate and display the accuracy score on the training data
            train_accuracy = accuracy_score(y_train, DT.predict(X_train))
            st.write("Training Accuracy:", train_accuracy)

    # Calculate and display the accuracy score on the test data
            dt_acc = accuracy_score(y_test, y_pred)
            st.write("Test Accuracy:", dt_acc)

    # Display the confusion matrix
            st.write("Confusion Matrix:")
            st.write(confusion_matrix(y_test, y_pred))

    # Display the classification report
            classification_rep = classification_report(y_test, y_pred)
            st.text(classification_rep)

    # Define the grid of hyperparameters for grid search
            grid_param = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 5, 7, 10],
        'splitter': ['best', 'random'],
        'min_samples_leaf': [1, 2, 3, 5, 7],
        'min_samples_split': [1, 2, 3, 5, 7],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    # Perform grid search using cross-validation
            grid_search_dt = GridSearchCV(DT, grid_param, cv=5, n_jobs=-1, verbose=1)
            grid_search_dt.fit(X_train, y_train)

    # Print the best parameters and best score found during grid search
            st.write("Best Parameters:", grid_search_dt.best_params_)
            st.write("Best Score:", grid_search_dt.best_score_)

    # Get the best decision tree model
            DT = grid_search_dt.best_estimator_

    # Make predictions on the test data using the best model
            y_pred = DT.predict(X_test)

    # Calculate and display the accuracy score on the training data using the best model
            train_accuracy_best = accuracy_score(y_train, DT.predict(X_train))
            st.write("Training Accuracy (Best Model):", train_accuracy_best)

    # Calculate and display the accuracy score on the test data using the best model
            dt_acc_best = accuracy_score(y_test, y_pred)
            st.write("Test Accuracy (Best Model):", dt_acc_best)

    # Display the confusion matrix using the best model
            st.write("Confusion Matrix (Best Model):")
            st.write(confusion_matrix(y_test, y_pred))

    # Display the classification report using the best model
            classification_rep_best = classification_report(y_test, y_pred)
            st.text(classification_rep_best)
            
        if st.button("Random Forest"):
    # Define the Random Forest classifier with specified hyperparameters
            rand_clf = RandomForestClassifier(criterion='entropy', max_depth=15, max_features=0.75,
                                       min_samples_leaf=2, min_samples_split=3,
                                       n_estimators=130)

    # Fit the Random Forest classifier to the training data
            rand_clf.fit(X_train, y_train)

    # Make predictions on the test data
            y_pred = rand_clf.predict(X_test)

    # Calculate and display the accuracy score on the training data
            train_accuracy = accuracy_score(y_train, rand_clf.predict(X_train))
            st.write("Training Accuracy:", train_accuracy)

    # Calculate and display the accuracy score on the test data
            rand_acc = accuracy_score(y_test, y_pred)
            st.write("Test Accuracy:", rand_acc)

    # Display the confusion matrix
            st.write("Confusion Matrix:")
            st.write(confusion_matrix(y_test, y_pred))

    # Display the classification report
            classification_rep = classification_report(y_test, y_pred)
            st.text(classification_rep)
          
        if st.button("Gradient Boosting Classifier"):
    # Define the Gradient Boosting classifier with specified hyperparameters
            gbc = GradientBoostingClassifier(learning_rate=0.1, loss='exponential', n_estimators=150)

    # Fit the Gradient Boosting classifier to the training data
            gbc.fit(X_train, y_train)

    # Make predictions on the test data
            y_pred = gbc.predict(X_test)

    # Calculate and display the accuracy score on the training data
            train_accuracy = accuracy_score(y_train, gbc.predict(X_train))
            st.write("Training Accuracy:", train_accuracy)

    # Calculate and display the accuracy score on the test data
            gbc_acc = accuracy_score(y_test, y_pred)
            st.write("Test Accuracy:", gbc_acc)

    # Display the confusion matrix
            st.write("Confusion Matrix:")
            st.write(confusion_matrix(y_test, y_pred))

    # Display the classification report
            classification_rep = classification_report(y_test, y_pred)
            st.text(classification_rep)
            
        if st.button("XGBoost Classifier"):
    # Define the XGBoost classifier with specified hyperparameters
            xgb = XGBClassifier(objective='binary:logistic', learning_rate=0.01, max_depth=10, n_estimators=180)

    # Fit the XGBoost classifier to the training data
            xgb.fit(X_train, y_train)

    # Make predictions on the test data
            y_pred = xgb.predict(X_test)

    # Calculate and display the accuracy score on the training data
            train_accuracy = accuracy_score(y_train, xgb.predict(X_train))
            st.write("Training Accuracy:", train_accuracy)

    # Calculate and display the accuracy score on the test data
            xgb_acc = accuracy_score(y_test, y_pred)
            st.write("Test Accuracy:", xgb_acc)

    # Display the confusion matrix
            st.write("Confusion Matrix:")
            st.write(confusion_matrix(y_test, y_pred))

    # Display the classification report
            classification_rep = classification_report(y_test, y_pred)
            st.text(classification_rep)                         
       
                            
    if choices == 'PREDICTION':
        st.subheader("PREDICTION")
        
        Preg = st.selectbox('Select Pregnancy count',tuple(Pregnancies.keys()))
        Glu = st.selectbox('Select Glucose Level',tuple(Glucose.keys()))
        BP = st.selectbox('Select Blood Pressure Range',tuple(BloodPressure.keys()))
        ST = st.selectbox('Skin Thickness',tuple(SkinThickness.keys()))
        Ins = st.selectbox('Insulin Level',tuple(Insulin.keys()))
        Body_mass = st.selectbox('Select Body Mass Index Range',tuple(BMI.keys()))
        DPF = st.selectbox('Diabetes Pedigree Function',tuple(DiabetesPedigreeFunction.keys()))
        Age_range = st.selectbox('AGE',tuple(Age.keys()))
       
    
        v_Preg = get_value(Preg,Pregnancies)
        v_Glu = get_value(Glu,Glucose)
        v_BP = get_value(BP,BloodPressure)
        v_ST = get_value(ST,SkinThickness)
        v_Ins = get_value(Ins,Insulin)
        v_Body_mass = get_value(Body_mass,BMI)
        v_DPF = get_value(DPF,DiabetesPedigreeFunction)
        v_Age_range = get_value(Age_range,Age)
    
        
        print("v_Preg:", v_Preg)
        print("v_Glu:", v_Glu)
        print("v_BP:", v_BP)
        print("v_ST:", v_ST)
        print("v_Ins:", v_Ins)
        print("v_Body_mass:", v_Body_mass)
        print("v_DPF:", v_DPF)
        print("v_Age_range:", v_Age_range)
      
    
        
        pretty_data = {
            "Preg":Preg,
            "Glu":Glu,
            "BP":BP,
            "ST":ST,
            "Ins":Ins,
            "Body_mass":Body_mass,
            "DPF":DPF,
            "Age_range":Age_range,
            
        }
        st.subheader("Options Selected")
        st.json(pretty_data)
        
        st.subheader("Data Encoded As")
        sample_data = [v_Preg,v_Glu,v_BP,v_ST,v_Ins,v_Body_mass,v_DPF,v_Age_range]
        st.write(sample_data)
        
        prep_data = np.array(sample_data).reshape(1,-1)
        
        model_choice = st.selectbox("Model Choice",["Logistic Regression", "KNN", "SVM", "Decision Tree Classifier", "Random Forest Classifier", "Gradient Boosting Classifier", "XgBoost"])
        
        if st.button("Evaluate"):
            if model_choice == 'Logistic Regression':
                predictor = load_prediction_model("models/logistic_regression_model.pkl")
                prediction = predictor.predict(prep_data)
                
                pred_prob_diabetic = predictor.predict_proba(prep_data)[:, 1]  # Probability of being diabetic
                pred_prob_not_diabetic = 1 - pred_prob_diabetic  # Probability of not being diabetic
                pred_prob_diabetic_percentage = pred_prob_diabetic[0] * 100
                pred_prob_not_diabetic_percentage = pred_prob_not_diabetic[0] * 100
                
                st.write(f"Probability of being diabetic: {pred_prob_diabetic_percentage:.2f}%")
                st.write(f"Probability of not being diabetic: {pred_prob_not_diabetic_percentage:.2f}%")
                final_result = get_key(prediction, Outcome)
                
            elif model_choice == 'KNN':
                predictor = load_prediction_model("models/KNN.pkl")
                prediction = predictor.predict(prep_data)
                pred_prob_diabetic = predictor.predict_proba(prep_data)[:, 1]  
                pred_prob_not_diabetic = 1 - pred_prob_diabetic  
                pred_prob_diabetic_percentage = pred_prob_diabetic[0] * 100
                pred_prob_not_diabetic_percentage = pred_prob_not_diabetic[0] * 100

                st.write(f"Probability of being diabetic: {pred_prob_diabetic_percentage:.2f}%")
                st.write(f"Probability of not being diabetic: {pred_prob_not_diabetic_percentage:.2f}%")

                final_result = get_key(prediction, Outcome)
                
            elif model_choice == 'SVM':
                predictor = load_prediction_model("models/SVM.pkl")
                prediction = predictor.predict(prep_data)
                final_result = get_key(prediction, Outcome)  
            
            elif model_choice == 'Decision Tree Classifier':
                predictor = load_prediction_model("models/DecisionTree.pkl")
                prediction = predictor.predict(prep_data)
                pred_prob_diabetic = predictor.predict_proba(prep_data)[:, 1]  
                pred_prob_not_diabetic = 1 - pred_prob_diabetic  
                pred_prob_diabetic_percentage = pred_prob_diabetic[0] * 100
                pred_prob_not_diabetic_percentage = pred_prob_not_diabetic[0] * 100

                st.write(f"Probability of being diabetic: {pred_prob_diabetic_percentage:.2f}%")
                st.write(f"Probability of not being diabetic: {pred_prob_not_diabetic_percentage:.2f}%")

                final_result = get_key(prediction, Outcome)
            elif model_choice == 'Random Forest Classifier':
                predictor = load_prediction_model("models/RandomForest.pkl")
                prediction = predictor.predict(prep_data)
                pred_prob_diabetic = predictor.predict_proba(prep_data)[:, 1]  
                pred_prob_not_diabetic = 1 - pred_prob_diabetic  
                pred_prob_diabetic_percentage = pred_prob_diabetic[0] * 100
                pred_prob_not_diabetic_percentage = pred_prob_not_diabetic[0] * 100

                st.write(f"Probability of being diabetic: {pred_prob_diabetic_percentage:.2f}%")
                st.write(f"Probability of not being diabetic: {pred_prob_not_diabetic_percentage:.2f}%")

                final_result = get_key(prediction, Outcome)
                
            
            elif model_choice == 'Gradient Boosting Classifier':
                predictor = load_prediction_model("models/GradientBoosting.pkl")
                prediction = predictor.predict(prep_data)
                pred_prob_diabetic = predictor.predict_proba(prep_data)[:, 1]  
                pred_prob_not_diabetic = 1 - pred_prob_diabetic  
                pred_prob_diabetic_percentage = pred_prob_diabetic[0] * 100
                pred_prob_not_diabetic_percentage = pred_prob_not_diabetic[0] * 100

                st.write(f"Probability of being diabetic: {pred_prob_diabetic_percentage:.2f}%")
                st.write(f"Probability of not being diabetic: {pred_prob_not_diabetic_percentage:.2f}%")

                final_result = get_key(prediction, Outcome)
                
            
            elif model_choice == 'XgBoost':
                predictor = load_prediction_model("models/XGBoost.pkl")
                prediction = predictor.predict(prep_data)
                pred_prob_diabetic = predictor.predict_proba(prep_data)[:, 1]  
                pred_prob_not_diabetic = 1 - pred_prob_diabetic  
                pred_prob_diabetic_percentage = pred_prob_diabetic[0] * 100
                pred_prob_not_diabetic_percentage = pred_prob_not_diabetic[0] * 100

                st.write(f"Probability of being diabetic: {pred_prob_diabetic_percentage:.2f}%")
                st.write(f"Probability of not being diabetic: {pred_prob_not_diabetic_percentage:.2f}%")

                final_result = get_key(prediction, Outcome)
                
                
                
            final_result = get_key(prediction,Outcome)
            if prediction[0] == 1:
             st.error(f"The person is diabetic.")
            else:
             st.success(f"The person is not diabetic")
        
    if choices == 'ABOUT':
        st.subheader("ABOUT DIABETES")

        st.write("Diabetes is a chronic condition characterized by elevated levels of blood glucose (sugar). It occurs when the body either doesn't produce enough insulin or can't effectively use the insulin it produces. Insulin is a hormone produced by the pancreas that helps glucose enter cells to be used for energy.")

        st.write("There are several types of diabetes, including type 1, type 2, and gestational diabetes. Each type has its own causes, risk factors, and management strategies.")

        st.write("Common symptoms of diabetes include frequent urination, excessive thirst, unexplained weight loss, fatigue, blurred vision, and slow wound healing.")

        st.write("Diabetes management typically involves lifestyle changes such as a healthy diet, regular exercise, monitoring blood sugar levels, and, in some cases, medication or insulin therapy.")

        st.write("Complications of diabetes can include heart disease, stroke, kidney disease, nerve damage, and vision problems. However, with proper management and lifestyle modifications, many people with diabetes can lead healthy and fulfilling lives.")

        st.write("It's important for individuals with diabetes to work closely with their healthcare team to develop a personalized treatment plan and to regularly monitor their blood sugar levels.")

        st.write("For more information about diabetes, you can visit reputable sources such as the American Diabetes Association (ADA) or the Centers for Disease Control and Prevention (CDC).")
        
        st.subheader("Precautions for Type 1 Diabetes:")
        st.write("- Take insulin as prescribed by your doctor.")
        st.write("- Monitor blood sugar levels regularly and adjust insulin doses accordingly.")
        st.write("- Follow a balanced diet, paying attention to carbohydrate intake.")
        st.write("- Engage in regular physical activity to help control blood sugar levels.")
        st.write("- Carry fast-acting carbohydrates, such as glucose tablets, in case of hypoglycemia (low blood sugar).")

        st.subheader("Precautions for Type 2 Diabetes:")
        st.write("- Follow a healthy diet that is low in sugar, saturated fats, and processed foods.")
        st.write("- Engage in regular physical activity, aiming for at least 30 minutes of moderate exercise most days of the week.")
        st.write("- Monitor blood sugar levels regularly, especially if you are on medication.")
        st.write("- Maintain a healthy weight or work towards achieving a healthy weight if overweight or obese.")
        st.write("- Be mindful of portion sizes and avoid overeating.")
if __name__ == '__main__':
    main()
  
    