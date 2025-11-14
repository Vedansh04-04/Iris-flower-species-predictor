import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
@st.cache_data
def load_data():
    iris = load_iris()
    print(iris)
    df = pd.DataFrame(data = iris.data,columns = iris.feature_names)
    df.columns =['sepal_length','sepal_width','petal_length','petal_width']
    df['target'] = iris.target
    df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    return df, iris.target_names

df,species_names = load_data()

st.title("Iris Species Classifier using KNN")

st.text("This interactive app demonstrates the KNN algorithm to classify iris flower based on user provided measurements")

st.sidebar.title("User Input Features")

st.image("https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Machine+Learning+R/iris-machinelearning.png",caption="Iris setosa, one of the species classified.")

def user_input_features():
    sepal_length = st.sidebar.slider("sepal length:", float(df['sepal_length'].min()),float(df['sepal_length'].max()), 5.40)
    sepal_width = st.sidebar.slider("sepal width:",float(df['sepal_width'].min()),float(df['sepal_width'].max()), 3.40)
    petal_length = st.sidebar.slider("petal length:", float(df['petal_length'].min()),float(df['petal_length'].max()), 1.30)
    petal_width = st.sidebar.slider("petal width:", float(df['sepal_width'].min()),float(df['sepal_width'].max()), 0.20)
    k_value = st.sidebar.slider("number of neighbors:", 1, 15, 5, step=2)

    data = {'sepal_length':sepal_length,
            'sepal_width':sepal_width,
            'petal_length':petal_length,
            'petal_width':petal_width}

    features = pd.DataFrame(data, index =[0])
    return features, k_value

input_df, k = user_input_features()
st.subheader("User Input Parameters")
st.write(input_df)

X = df[['sepal_length','sepal_width','petal_length','petal_width']]
y = df['target']

X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.2,random_state=42)


knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train,y_train)
prediction = knn.predict(input_df)
prediction_proba = knn.predict_proba(input_df)
print(prediction)

st.subheader("prediction")
st.write(f"the predicted species is : **{species_names[prediction[0]]}**")
st.write(f"this is based on the **{k}** nearest neighbors")

st.subheader("Prediction Probability")
prob_df = pd.DataFrame(prediction_proba, columns = species_names)
st.bar_chart(prob_df.T)

st.subheader("Mmodel Accuracy")
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
st.write(f"the model's accuracy on the test set is: **{accuracy:.2f}**")