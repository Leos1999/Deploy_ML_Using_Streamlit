import streamlit as st
from sklearn import datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("ML Using Streamlit")

# st.write("""
# # Explore the world
# hello world
# """)

dataset = st.sidebar.selectbox("Select dataset",("Iris Dataset","Cancer Dataset","Wine Dataset"))
st.write("The Dataset selected is:",dataset)

classifier_name = st.sidebar.selectbox("Select classifier",("KNN","SVM","Random Forest"))
st.write("The Classifier selected is:",classifier_name)

def get_dataset(dataset):
    if dataset == 'Iris Dataset':
        data = datasets.load_iris()
    elif dataset == 'Cancer Dataset':
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    x = data.data
    y = data.target
    return x,y
    
x,y = get_dataset(dataset)
st.write("Shape of dataset",x.shape)
st.write("No.of classes",len(np.unique(y)))

def add_param_UI(clf_name):
    params = {}
    if clf_name == 'KNN':
        K = st.sidebar.slider("K",1,15)
        params["K"] = K
    elif clf_name == 'SVM':
        C = st.sidebar.slider("C",0.01,10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("Max_Depth",2,15)
        n_estimators = st.sidebar.slider("n_estimators",1,100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params 

params = add_param_UI(classifier_name)

def get_classifier(clf_name,param):
    if clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=param["K"])
    elif clf_name == 'SVM':
        clf = SVC(C=param["C"])
    else:
        clf = RandomForestClassifier(n_estimators=param["n_estimators"],max_depth=param["max_depth"],random_state=2)
    return clf

clf = get_classifier(classifier_name,params)

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=3)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test,y_pred)
st.write(f"classifier={classifier_name}")
st.write(f"Accuracy ={acc}")

#Plot
pca = PCA(2)
X_projected = pca.fit_transform(x)

x1 = X_projected[:,0]
x2 = X_projected[:,1]

fig = plt.figure()
plt.scatter(x1,x2,c=y,alpha=0.8,cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

#plt.show()
st.pyplot()