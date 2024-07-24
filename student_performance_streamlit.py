import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import warnings
import io

warnings.filterwarnings("ignore")

# Load and display data
st.title('Student Performance Analysis')

@st.cache
def load_data():
    df = pd.read_csv("StudentsPerformance.csv")
    return df

df = load_data()

if st.checkbox('Show raw data'):
    st.write(df)

# Show data information
st.subheader('Data Information')
if st.checkbox('Show data info'):
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

# Show description
if st.checkbox('Show data description'):
    st.write(df.describe())

# Show unique categories in columns
st.subheader('Unique Categories in Columns')
if st.checkbox('Show unique categories'):
    st.write(f"Categories in 'gender': {df['gender'].unique()}")
    st.write(f"Categories in 'race/ethnicity': {df['race/ethnicity'].unique()}")
    st.write(f"Categories in 'parental level of education': {df['parental level of education'].unique()}")
    st.write(f"Categories in 'lunch': {df['lunch'].unique()}")
    st.write(f"Categories in 'test preparation course': {df['test preparation course'].unique()}")

# Add derived columns
df['total score'] = df['math score'] + df['reading score'] + df['writing score']
df['average'] = df['total score'] / 3

# Plots
st.subheader('Distribution of Average Scores')
fig, axs = plt.subplots(1, 2, figsize=(15, 7))
sns.histplot(data=df, x='average', bins=30, kde=True, color='g', ax=axs[0])
sns.histplot(data=df, x='average', kde=True, hue='gender', ax=axs[1])
st.pyplot(fig)

st.subheader('Distribution of Total Scores')
fig, axs = plt.subplots(1, 2, figsize=(15, 7))
sns.histplot(data=df, x='total score', bins=30, kde=True, color='g', ax=axs[0])
sns.histplot(data=df, x='total score', kde=True, hue='gender', ax=axs[1])
st.pyplot(fig)

# Analysis based on lunch
st.subheader('Average Scores by Lunch Type')
fig, axs = plt.subplots(1, 3, figsize=(25, 6))
sns.histplot(data=df, x='average', kde=True, hue='lunch', ax=axs[0])
sns.histplot(data=df[df.gender == 'female'], x='average', kde=True, hue='lunch', ax=axs[1])
sns.histplot(data=df[df.gender == 'male'], x='average', kde=True, hue='lunch', ax=axs[2])
st.pyplot(fig)

# Analysis based on race/ethnicity
st.subheader('Average Scores by Race/Ethnicity')
fig, axs = plt.subplots(1, 3, figsize=(25, 6))
sns.histplot(data=df, x='average', kde=True, hue='race/ethnicity', ax=axs[0])
sns.histplot(data=df[df.gender == 'female'], x='average', kde=True, hue='race/ethnicity', ax=axs[1])
sns.histplot(data=df[df.gender == 'male'], x='average', kde=True, hue='race/ethnicity', ax=axs[2])
st.pyplot(fig)

# Pie charts
st.subheader('Distribution of Features')
fig, axs = plt.subplots(1, 5, figsize=(30, 12))
sizes = [df['gender'].value_counts(), df['race/ethnicity'].value_counts(), df['lunch'].value_counts(), df['test preparation course'].value_counts(), df['parental level of education'].value_counts()]
labels = ['Female', 'Male'], ['Group C', 'Group D', 'Group B', 'Group E', 'Group A'], ['Standard', 'Free'], ['None', 'Completed'], ['Some College', "Associate's Degree", 'High School', 'Some High School', "Bachelor's Degree", "Master's Degree"]
colors = ['red', 'green'], ['red', 'green', 'blue', 'cyan', 'orange'], ['red', 'green'], ['red', 'green'], ['red', 'green', 'blue', 'cyan', 'orange', 'grey']

for i in range(5):
    axs[i].pie(sizes[i], colors=colors[i], labels=labels[i], autopct='.2f%%')
    axs[i].set_title(['Gender', 'Race/Ethnicity', 'Lunch', 'Test Course', 'Parental Education'][i], fontsize=20)
    axs[i].axis('off')

st.pyplot(fig)

# Bivariate analysis
st.subheader('Bivariate Analysis')
st.write("Analyzing the impact of race/ethnicity and test preparation course on scores.")

# Race/Ethnicity impact
fig, ax = plt.subplots(1, 3, figsize=(20, 8))
sns.barplot(x=df.groupby('race/ethnicity')['math score'].mean().index, y=df.groupby('race/ethnicity')['math score'].mean().values, palette='mako', ax=ax[0])
sns.barplot(x=df.groupby('race/ethnicity')['reading score'].mean().index, y=df.groupby('race/ethnicity')['reading score'].mean().values, palette='flare', ax=ax[1])
sns.barplot(x=df.groupby('race/ethnicity')['writing score'].mean().index, y=df.groupby('race/ethnicity')['writing score'].mean().values, palette='coolwarm', ax=ax[2])
st.pyplot(fig)

# Test preparation impact
fig, axs = plt.subplots(2, 2, figsize=(12, 6))
sns.barplot(x=df['lunch'], y=df['math score'], hue=df['test preparation course'], ax=axs[0, 0])
sns.barplot(x=df['lunch'], y=df['reading score'], hue=df['test preparation course'], ax=axs[0, 1])
sns.barplot(x=df['lunch'], y=df['writing score'], hue=df['test preparation course'], ax=axs[1, 0])
st.pyplot(fig)

# ML Model Training
st.subheader('Machine Learning Model Training')

X = df.drop(columns=['math score'], axis=1)
y = df['math score']

num_features = X.select_dtypes(exclude="object").columns
cat_features = X.select_dtypes(include="object").columns

numeric_transformer = StandardScaler()
oh_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    [
        ("OneHotEncoder", oh_transformer, cat_features),
        ("StandardScaler", numeric_transformer, num_features),
    ]
)

X = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, mse, rmse, r2_square

models = {
    
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "AdaBoost Regressor": AdaBoostRegressor()
}

model_list = []
r2_list = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    model_train_mae, model_train_mse, model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)
    model_test_mae, model_test_mse, model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)

    st.write(f"{name}")
    st.write('Model performance for Training set')
    st.write(f"- Root Mean Squared Error: {model_train_rmse:.4f}")
    st.write(f"- Mean Squared Error: {model_train_mse:.4f}")
    st.write(f"- Mean Absolute Error: {model_train_mae:.4f}")
    st.write(f"- R2 Score: {model_train_r2:.4f}")

    st.write('Model performance for Test set')
    st.write(f"- Root Mean Squared Error: {model_test_rmse:.4f}")
    st.write(f"- Mean Squared Error: {model_test_mse:.4f}")
    st.write(f"- Mean Absolute Error: {model_test_mae:.4f}")
    st.write(f"- R2 Score: {model_test_r2:.4f}")

    model_list.append(name)
    r2_list.append(model_test_r2)

st.write(pd.DataFrame(list(zip(model_list, r2_list)), columns=['Model Name', 'R2_Score']).sort_values(by=["R2_Score"], ascending=False))

# Best model
lin_model = RandomForestRegressor()
lin_model.fit(X_train, y_train)
y_pred = lin_model.predict(X_test)
score = r2_score(y_test, y_pred) * 100
st.write(f"Accuracy of the model is {score:.2f}%")

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
sns.regplot(x=y_test, y=y_pred, ci=None, color='red', ax=ax)
st.pyplot(fig)

pred_df = pd.DataFrame({'Actual Value': y_test, 'Predicted Value': y_pred, 'Difference': y_test - y_pred})
if st.checkbox('Show prediction data'):
    st.write(pred_df)
