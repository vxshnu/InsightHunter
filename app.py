import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
#from streamlit_pandas_profiling import st_profile_report
import os
from sklearn.impute import SimpleImputer
import h2o
from h2o.automl import H2OAutoML
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import requests
import atexit

BACKEND_URL = "http://127.0.0.1:5000/predict"

block_select_box = False
if "regression_button" not in st.session_state:
    st.session_state.regression_button = False
if "classification_button" not in st.session_state:
    st.session_state.classification_button = False

if os.path.exists("data.csv"):
    df = pd.read_csv("data.csv")

def data_profiling():
    st.title("Data Report")
    profile = ProfileReport(df, title="Data Profiling Report", explorative=True)
    if st.checkbox('Show Dataset'):
        st.dataframe(df)
    st.write("")
    if st.checkbox('Show Data Profiling'):
        report_html = profile.to_html()
        st.components.v1.html(report_html, height=1000, scrolling=True)
    st.write("")
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
    st.title("INSIGHT :green[HUNTER]")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.subheader(":blue[_Clean and prepare your data effortlessly._]",divider=True)
    st.subheader(":green[_Explore trends with interactive visualizations._]",divider=True)
    st.subheader(":orange[_Build predictive models with ease._]",divider=True)
    st.subheader(":red[_Use AI to simplify data filtering and aggregation._]",divider=True)
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.caption("_Made by Vishnu Narayanan_")

def train_the_model(option_selected,type):
    global block_select_box
    st.info("The dataset should exclude the target column.")
    file2 = st.file_uploader("Model Prediction Data",type=['json','csv','xlsx'])
    if file2 is not None:
        if file2.name.endswith('.json'):
            df2 = pd.read_json(file2)
        elif file2.name.endswith('.csv'):
            df2 = pd.read_csv(file2)
        elif file2.name.endswith('.xlsx'):
            df2 = pd.read_excel(file2) 
    if option_selected:
        if st.button("Train the Model!"): 
            with st.spinner('Loading the model!'):
                h2o.init()
                h2o_df = h2o.H2OFrame(df)
                if type == 1:
                    h2o_df[option_selected] = h2o_df[option_selected].asfactor()
                else:
                    h2o_df[option_selected] = h2o_df[option_selected].asnumeric()
                train, test = h2o_df.split_frame(ratios=[0.01], seed=1234)
                aml = H2OAutoML(max_models=10, seed=1, max_runtime_secs=240)
            with st.spinner('Training the model... This may take a few minutes!'):
                aml.train(y=option_selected, training_frame=train)
            st.subheader("Model Statistics")
            leaderboard = aml.leaderboard
            st.write(leaderboard) 
            with st.spinner('Predicting...'):
                h2o_df2 = h2o.H2OFrame(df2)
                predicted_df = aml.predict(h2o_df2)
            predicted_values = predicted_df.as_data_frame()['predict']
            df2[option_selected] = predicted_values
            st.subheader("Dataset with predicted values")
            st.dataframe(df2)
            csv_file = "data_predict.csv"
            df2.to_csv(csv_file, index=False)
            with open("data_predict.csv", "r") as file:st.download_button(label="Download new generated CSV",data=file,file_name="data predict.csv",mime="text/csv")
            
            
def train_model():  
    st.title("Train Model")
    st.write("")
    left,right = st.columns(2)
    if left.button("Regression Model",use_container_width=True) :
        st.session_state.regression_button = True
        st.session_state.classification_button = False
    if right.button("Classification Model",use_container_width=True):
        st.session_state.regression_button = False
        st.session_state.classification_button = True
        
        
def analysis(df,key):
    st.write(df.head())
    st.subheader("Correlation Heatmap")
    heat_map_options = st.multiselect("Select columns for correlation",df.columns.tolist(),key=key+1)
    if heat_map_options:
        corr = df[heat_map_options].corr() 
        fig1 = plt.figure(figsize=(16, 14))  
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        st.pyplot(fig1)
    st.subheader("Line Plot")
    line_plot_options = st.multiselect("Select any 2 columns for line plot",df.columns.tolist(),max_selections=2,key=key+2)
    if len(line_plot_options) == 2:
        fig3, axes = plt.subplots(len(df[line_plot_options].columns), 1, figsize=(16, 5 * len(df[line_plot_options].columns)))
        for i, column in enumerate(df[line_plot_options].columns):
            axes[i].plot(df[column], label=column)
            axes[i].set_title(f"Line Plot for {column}")
            axes[i].legend()
        st.pyplot(fig3)
    st.subheader("Interactive Scatter Plot")
    scatter_plot_options = st.multiselect("Select any 2 columns for scatter plot",df.columns.tolist(),max_selections=2,key=key)
    if len(scatter_plot_options) == 2:
        fig4 = px.scatter_matrix(df[scatter_plot_options], height=1000, width=1200)  
        st.plotly_chart(fig4)


def checkForConnection():
    try:
        with st.spinner("Connecting to the server, please wait..."):
            response = requests.get(BACKEND_URL,timeout=5)
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.exceptions.RequestException:
        return False
        

def data_filtering_aggregation():
    st.info("An AI model capable of filtering and aggregating your data. Type in the text box your preferred filtering/aggregation criteria. Remember the model can make mistakes!")
    query = st.text_input("How can I help?",placeholder="Your question?",value = None)
    column_names = df.columns.tolist()
    if query is not None:
        query+=" (columns:"
        for i in range(0,len(column_names)):
            query+=" "
            if i == len(column_names)-1:
                query+=column_names[i]+")"
            else:
                query+=column_names[i]+","
        if checkForConnection():
            response = requests.post(BACKEND_URL, json={"query": query})
            print(response)
            code = response.json()    
            code['result'] = st.text_input("Type in your preferred code!",value = code['result'])
            st.info("If there’s an issue with the code, please provide the corrected version; otherwise, leave it unchanged.")
            if st.button("Show Result?"):
                local_variables = {"df": df} 
                exec("df3="+code['result'], {"__builtins__": None}, local_variables)
                df3 = local_variables.get("df3")
                if isinstance (df3,pd.DataFrame):
                    st.dataframe(df3)
                    df3.to_csv("filtered_csv.csv", index=False)
                    with open("filtered_csv.csv", "r") as file:st.download_button(label="Download new filtered dataset",data=file,file_name="data predict.csv",mime="text/csv")
                else:
                    st.write(df3)
        else:
            st.error("We're sorry, the server is currently unavailable. Please try again later.")

with st.sidebar:
    st.image("resources/photo-1666875753105-c63a6f3bdc86.jpg")
    st.title("INSIGHT HUNTER")
    main_choice = st.radio("Choices",['Home','Upload File','Data Profiling','Train Model','Visualization','Data Filtering/Aggregation'],label_visibility='hidden')

if main_choice == 'Home':
    st.session_state.regression_button = False
    st.session_state.classification_button = False
    redirect_to_home()
elif main_choice == 'Upload File':
    st.session_state.regression_button = False
    st.session_state.classification_button = False
    read_file_from_user()
elif main_choice == 'Data Profiling':
    if os.path.exists('data.csv'):
        st.session_state.regression_button = False
        st.session_state.classification_button = False
        data_profiling()
    else:
        st.warning("Upload a dataset to view insights!")
    
elif main_choice == 'Train Model':
    if os.path.exists('data.csv'):
        train_model()
    else:
        st.warning("Upload a dataset to view insights!")
elif main_choice == 'Visualization':
    if os.path.exists('data.csv'):
        st.session_state.regression_button = False
        st.session_state.classification_button = False
        st.title("Data Overview")
        st.subheader("Uploaded Dataset")
        analysis(df,20)
        if os.path.exists("data_predict.csv"):
            st.subheader("Prediction Generated Dataset")
            df2 = pd.read_csv("data_predict.csv")
            analysis(df2,10)
    else:
        st.warning("Upload a dataset to view insights!")
elif main_choice == 'Data Filtering/Aggregation':
    if os.path.exists('data.csv'):
        st.session_state.regression_button = False
        st.session_state.classification_button = False
        st.title('Data Filtering/Aggregation')
        data_filtering_aggregation()
    else:
        st.warning("Upload a dataset to view insights!")
    
if st.session_state.regression_button:
    option_selected = st.selectbox("The column you want to predict?",options = df.columns,disabled=block_select_box,)
    train_the_model(option_selected,0)
    
if st.session_state.classification_button:
    option_selected = st.selectbox("The column you want to classify?",options = df.columns,disabled=block_select_box,)
    train_the_model(option_selected,1)
    
def delete_files():
    paths = ['data.csv','data_predict.csv','filtered_csv.csv']
    for path in paths:
        if os.path.exists(path):
            os.remove(path)

atexit.register(delete_files)