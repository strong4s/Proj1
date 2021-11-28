import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static
import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from sklearn.preprocessing import Normalizer, MinMaxScaler
from math import pi
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from PIL import Image

pd.set_option('display.float_format', '{:,.4f}'.format)
pd.set_option('max_columns', 100)

# Importing Data and Data Cleaning
df=pd.read_csv("data/merged.csv")
df=df.loc[:, ~df.columns.str.contains("Twitter")]
df=df.loc[:, ~df.columns.str.contains("Facebook")]
for col in df.iloc[:,2:5].columns:
    df[col]=df[col].replace(np.nan, "")
for col in df.iloc[:,5:].columns:
    df[col]=df[col].replace(np.nan, 0)

# Feature Engineering
# Features for Expense Breakdown
df["Transportation and Communication"]=df["Travel Expenses"]+df["Communications"]
df["Labor"]=df["Compensation of campaigners, etc."] + df["Employment of Poll Watchers"] + df["Employment of Counsel"]
df["Supplies and Logistics"] = df["Stationery, Printing, and Distribution"] + df["Rent, Maintenance, etc."] + df["Political Meetings and Rallies"] + df["Copying and Classifying List of Voters"] + df["Printing of Sample Ballots"]
# Features for Contributions by Source
df["Contributions from Other Sources"] = df["Cash Contributions Received from Other Sources"] + df["In-Kind Contributions Received from Other Sources"]
df["Contributions from Political Party"] = df["Cash Contributions Received from Political Party"] + df["In-Kind Contributions Received from Political Party"]
# Features for Contributions by Type
df["Cash Contributions"] = df["Cash Contributions Received from Other Sources"] + df["Cash Contributions Received from Political Party"]
df["In-Kind Contributions"] = df["In-Kind Contributions Received from Other Sources"] +df["In-Kind Contributions Received from Political Party"]


my_page = st.sidebar.radio('Page Navigation', ['Homepage', 'Data', 'Contributions', 'Heat Maps', 'Spider Maps'])

if my_page == 'Homepage':
    #st.write("Powerpoint Presentation")
    
    image = Image.open('Sprint 1 Project - Group 3 Snowball.pptx.jpg')

    st.image(image, caption='Sunrise by the mountains')
    
elif my_page == 'Data':
    st.title("Data")
    st.header("2019 Senatorial Campaign Spending Data")
    if st.checkbox('Show data', value = True):
        st.subheader('Data')
        data_load_state = st.text('Loading data...')
        st.write(df)
        data_load_state.markdown('Loading data...**done!**')

elif my_page == 'Contributions':
    option_candidate = st.sidebar.selectbox('Which Senatorial Candidate Do You Want To See?', df['Candidate'].unique())
    option_contri = st.sidebar.selectbox('Contribution Type or Contribution Source?', ["Type", "Source"])
    
    'You selected contributions recieved by ', option_contri, 'of ', option_candidate

   
    contri_source = df[df['Candidate'] == option_candidate][["Contributions from Other Sources", "Contributions from Political Party"]]
    contri_type = df[df['Candidate'] == option_candidate][["Cash Contributions", "In-Kind Contributions"]]

    st.header(f"Contributions Received by {option_candidate}")
    
    if option_contri == "Type":
        plt.figure(figsize=(8,6))
        #fig = plt.figure(figsize=(8,6))
        
        #plt.bar(contri_type.transpose().index, contri_type.transpose().values)
        
        contri_type.transpose().plot(kind="bar")
         
        plt.title("Contributions Received by Type", fontsize=16)
        plt.ylabel("Contribution Amount", fontsize=12)
        plt.xlabel("Type of Contribution", fontsize=12)
        plt.xticks(rotation=45)
        plt.legend().remove()
        
        #for index, value in enumerate(contri_type.transpose().values()):
        #    plt.text(value, index, str(value))
    
        st.pyplot(plt)
    else:          
        plt.figure(figsize=(8,6))
        #fig= plt.figure(figsize=(8,6)) 
        #plt.bar(contri_source.transpose().index, contri_source.transpose().values)
        contri_source.transpose().plot(kind="bar")
        
        plt.title("Contributions Received by Source", fontsize=16)
        plt.ylabel("Contribution Amount", fontsize=12)
        plt.xlabel("Source of Contribution", fontsize=12)
        plt.xticks(rotation=45)
        plt.legend().remove()
        st.pyplot(plt)
        
elif my_page == 'Heat Maps':
    corr_type = st.sidebar.selectbox('Which Feature Correlation Do You Want to See?', ["Candidate Spending", "Contribution Type", "Contribution Source"])
    
    st.header(f"Correlation Heat Maps")
    
    if corr_type == "Candidate Spending":
        
        expenses_all = df.loc[:, "Votes":"Supplies and Logistics"]
        expenses_all["Pol Ads"] = df["Pol Ads"]

        plt.figure(figsize=(10, 8))

        labels=["Votes", "Transpo and Comms", "Labor", "Supplies and Logistics", "Pol Ads"]

        sns.heatmap(expenses_all.corr(), center=0.0, annot=True, xticklabels=labels, yticklabels=labels)
        plt.title("Spending  Breakdown Heat Map", fontsize=16)
        st.pyplot(plt)
        
    elif corr_type == "Contribution Type":
        
        contributions_type = df.loc[:, "Cash Contributions":"In-Kind Contributions"]
        contributions_type["Votes"]=df["Votes"]
        
        plt.figure(figsize=(10, 8))

        sns.heatmap(contributions_type.corr(), center=0.0, annot=True)
        plt.title("Contributions by Type", fontsize=16)

        st.pyplot(plt)
        
    else:
        contributions_source = df.loc[:, 'Contributions from Other Sources':"Contributions from Political Party"]
        contributions_source["Votes"]=df["Votes"]
        
        plt.figure(figsize=(10, 8))

        labels=["From Other Sources", "From Political Party", "Votes"]
        sns.heatmap(contributions_source.corr(), center=0.0, annot=True, xticklabels=labels, yticklabels=labels)
        plt.title("Contributions by Source", fontsize=16)

        st.pyplot(plt)
    
elif my_page == 'Spider Maps': 
    st.header(f"Clustering Spider Maps")
    
    # Select features for clustering
    feature_cols = ["Contributions from Other Sources",'Expenditures Paid Out of Cash Contributions','Pol Ads',"Supplies and Logistics",'Win']
    X = df[feature_cols]
    df_sample = df[feature_cols]
    # Feature scaling
    X=Normalizer().fit_transform(X.values)
    
    #fitting
    kmeans = KMeans(n_clusters=6, random_state=42)
    kmeans.fit(X)
    cluster_labels = kmeans.predict(X)
    df['Cluster Labels'] = cluster_labels
    
    
    kmeans = KMeans(n_clusters=6)
    kmeans.fit(X)
    y_pred = kmeans.predict(X)
    #y_pred
    
    #MinMax
    scaler = MinMaxScaler()
    df_minmax = scaler.fit_transform(df_sample)

    df_minmax = pd.DataFrame(df_minmax, index=df_sample.index, columns=df_sample.columns)

    df_minmax['Cluster_Labels'] = cluster_labels

    df_clusters = df_minmax.set_index("Cluster_Labels")
    df_clusters = df_clusters.groupby("Cluster_Labels").mean().reset_index()
    #df_clusters
    
    #Spider
    def make_spider(row, title, color):
 
        # number of variable
        categories=list(df_clusters)[1:]
        N = len(categories)

        # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        # Initialise the spider plot
        ax = plt.subplot(3,3,row+1, polar=True )

        # If you want the first axis to be on top:
        ax.set_theta_offset(pi / 3.5)
        ax.set_theta_direction(-1)

        # Draw one axe per variable + add labels labels yet
        plt.xticks(angles[:-1], categories, color='grey', size=8)

        # Draw ylabels
        ax.set_rlabel_position(0)
    #     plt.yticks([-2, -1, 0, 1, 2], [-2,-1, 0, 1, 2], color="grey", size=7) #for sscaled
    #     plt.ylim(-2.5,2.5)
        plt.yticks([-0.25, 0, 0.25, 0.5, 0.75, 1], [-0.25, 0, 0.25, 0.5,0.75, 1], color="grey", size=7) #formmscaled
        plt.ylim(-0.25,1)

        # Ind1
        values=df_clusters.loc[row].drop('Cluster_Labels').values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
        ax.fill(angles, values, color=color, alpha=0.4)

        # Add a title
        plt.title(title, size=14, color=color, y=1.1)
        #st.pyplot(plt)

    #Plotting Spider
    my_dpi=100
    plt.figure(figsize=(1000/my_dpi, 1000/my_dpi), dpi=my_dpi)
    plt.subplots_adjust(hspace=0.5)

    # Create a color palette:
    my_palette = plt.cm.get_cmap("Set2", len(df_clusters.index))

    for row in range(0, len(df_clusters.index)):
        make_spider(row=row, 
                    title='Segment '+(df_clusters['Cluster_Labels'][row]).astype(str), 
                    color=my_palette(row))  
    st.pyplot(plt)
    
