#myenv is the virtual environment name

import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import os
from sklearn.impute import SimpleImputer
from scipy import stats
import h2o
from h2o.automl import H2OAutoML
import time
from sklearn.model_selection import train_test_split

block_select_box = False
if "regression_button" not in st.session_state:
    st.session_state.regression_button = False
if "classification_button" not in st.session_state:
    st.session_state.classification_button = False

if os.path.exists("data.csv"):
    df = pd.read_csv("data.csv")

def data_profiling():
    st.title("Data Report")
    profile = ProfileReport(df, title="Data Profiling Report")
    if st.checkbox('Show Dataset'):
        st.dataframe(df)
    if st.checkbox('Show Data Profiling'):
        st_profile_report(profile)
    if st.checkbox('Perform Data Preprocessing'):
        flag = 0
        selections = st.multiselect("Methods Available",["Remove Rows with Null Values","Imputation","Removing Duplicates"])
        for selection in selections:
            flag = 1
            if selection == "Remove Rows with Null Values":
                count = df.isna().any(axis=1).sum()
                st.text(f"Total Number of Rows with Null Values : {count}")
                df.dropna(inplace=True)
                st.success("Successfully removed null values!")
            if selection == "Imputation":
                count = df.isna().any(axis=0).sum()
                st.text(f"Total Number of Columns with Null Values : {count}")
                num_cols = df.select_dtypes(include=['float64', 'int64']).columns
                cat_cols = df.select_dtypes(include=['object']).columns
                num_imputer = SimpleImputer(strategy='median')
                df[num_cols] = num_imputer.fit_transform(df[num_cols])
                cat_imputer = SimpleImputer(strategy='most_frequent')
                df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
                st.success("Successfully imputed null values!")
            if selection == "Removing Duplicates":
                count = df.duplicated().sum()
                st.text(f"Total Number of Duplicated Rows : {count}")
                df.drop_duplicates(inplace=True)
                st.success("Successfully deleted duplicates!")
        if flag == 1:       
            if st.button("Save updated dataset ?"):
                df.to_csv("data.csv", index=False)
                st.success("Successfully saved updated dataset!")
                with open("data.csv", "r") as file:st.download_button(label="Download CSV",data=file,file_name="dataset.csv",mime="text/csv")
        

def read_file_from_user():
    st.title("Upload Dataset File")
    st.info("The file format should be either JSON, CSV, or XLSX.")
    file = st.file_uploader("Dataset",type=['json','csv','xlsx'])
    if file is not None:
        if file.name.endswith('.json'):
            df = pd.read_json(file)
        elif file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        csv_file = "data.csv"
        df.to_csv(csv_file, index=False)

def redirect_to_home():
    pass

def regression():
    global block_select_box
    option_selected = st.selectbox("The column you want to predict?",options = df.columns,disabled=block_select_box,)
    if option_selected:
        if st.button("Train the Model!"):
            with st.spinner('Loading the model!'):
                h2o.init()
                h2o_df = h2o.H2OFrame(df)
                train, test = h2o_df.split_frame(ratios=[.8], seed=1234)
                aml = H2OAutoML(max_models=10, seed=1, max_runtime_secs=240)
            with st.spinner('Training the model... This may take a few minutes!'):
                aml.train(y=option_selected, training_frame=train)
            st.subheader("Leaderboard")
            leaderboard = aml.leaderboard
            st.write(leaderboard)

    
def classification():
    global block_select_box
    option_selected = st.selectbox("The column you want to predict?",options = df.columns,disabled=block_select_box,)
    if option_selected:
        if st.button("Train the Model!"):
            h2o.init()
            block_select_box = True

def perform_analysis():  
    left,right = st.columns(2)
    if left.button("Regression Model",use_container_width=True) :
        st.session_state.regression_button = True
        st.session_state.classification_button = False
    if right.button("Classification Model",use_container_width=True):
        st.session_state.regression_button = False
        st.session_state.classification_button = True

with st.sidebar:
    st.image("resources/photo-1666875753105-c63a6f3bdc86.jpg")
    st.title("INSIGHT HUNTER")
    main_choice = st.radio("Choices",['Home','Upload File','Data Profiling','Perform Analysis'],label_visibility='hidden')

if main_choice == 'Home':
    st.session_state.regression_button = False
    st.session_state.classification_button = False
    redirect_to_home()
elif main_choice == 'Upload File':
    st.session_state.regression_button = False
    st.session_state.classification_button = False
    read_file_from_user()
elif main_choice == 'Data Profiling':
    st.session_state.regression_button = False
    st.session_state.classification_button = False
    data_profiling()
elif main_choice == 'Perform Analysis':
    perform_analysis()
    
if st.session_state.regression_button:
    regression()
    
if st.session_state.classification_button:
    classification()