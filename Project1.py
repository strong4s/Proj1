import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static
# import geopandas as gpd
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


my_page = st.sidebar.radio('Page Navigation', ['Homepage', 'Data', 'Contributions', 'Expense Breakdown','Top Spenders','Heat Maps', 'Spider Maps'])

if my_page == 'Homepage':
    #st.write("Powerpoint Presentation")
    
    image = Image.open('Sprint 1 Project - Group 3 Snowball.pptx.jpg')
    image1 = Image.open('Sprint 1 Project - Group 3 Snowball.pptx (1).jpg')
    image2 = Image.open('Sprint 1 Project - Group 3 Snowball.pptx (2).jpg')
    image3 = Image.open('Sprint 1 Project - Group 3 Snowball.pptx (3).jpg')
    image4 = Image.open('Sprint 1 Project - Group 3 Snowball.pptx (4).jpg')
    image5 = Image.open('Sprint 1 Project - Group 3 Snowball.pptx (5).jpg')
    image6 = Image.open('Sprint 1 Project - Group 3 Snowball.pptx (6).jpg')
    image7 = Image.open('Sprint 1 Project - Group 3 Snowball.pptx (7).jpg')
    image8 = Image.open('Sprint 1 Project - Group 3 Snowball.pptx (8).jpg')
    image9 = Image.open('Sprint 1 Project - Group 3 Snowball.pptx (9).jpg')
    image10 = Image.open('Sprint 1 Project - Group 3 Snowball.pptx (10).jpg')
    image11 = Image.open('Sprint 1 Project - Group 3 Snowball.pptx (11).jpg')
    image12 = Image.open('Sprint 1 Project - Group 3 Snowball.pptx (12).jpg')
    image13 = Image.open('Sprint 1 Project - Group 3 Snowball.pptx (13).jpg')
    image14 = Image.open('Sprint 1 Project - Group 3 Snowball.pptx (14).jpg')
    image15 = Image.open('Sprint 1 Project - Group 3 Snowball.pptx (15).jpg')
    image16 = Image.open('Sprint 1 Project - Group 3 Snowball.pptx (16).jpg')
    image17 = Image.open('Sprint 1 Project - Group 3 Snowball.pptx (17).jpg')
    image18 = Image.open('Sprint 1 Project - Group 3 Snowball.pptx (18).jpg')
    image19 = Image.open('Sprint 1 Project - Group 3 Snowball.pptx (19).jpg')
    image20 = Image.open('Sprint 1 Project - Group 3 Snowball.pptx (20).jpg')
    image21 = Image.open('Sprint 1 Project - Group 3 Snowball.pptx (21).jpg')
    image22 = Image.open('Sprint 1 Project - Group 3 Snowball.pptx (22).jpg')
    image23 = Image.open('Sprint 1 Project - Group 3 Snowball.pptx (23).jpg')
    image24 = Image.open('Sprint 1 Project - Group 3 Snowball.pptx (24).jpg')
    image25 = Image.open('Sprint 1 Project - Group 3 Snowball.pptx (25).jpg')
    image26 = Image.open('Sprint 1 Project - Group 3 Snowball.pptx (26).jpg')
    image27 = Image.open('Sprint 1 Project - Group 3 Snowball.pptx (27).jpg')
    image28 = Image.open('Sprint 1 Project - Group 3 Snowball.pptx (28).jpg')
    image29 = Image.open('Sprint 1 Project - Group 3 Snowball.pptx (29).jpg')
    st.image([image, image1, image2, image3,image4,image5,image6,image7,image8,image9,image10,image11,image12,image13,image14,image15,image16,image17,image18,image19,image20,image21,image22,image23,image24,image25,image26,image27,image28,image29 ])
    #st.image(image1)
    #st.image(image2)
    
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
        
elif my_page == 'Expense Breakdown':
    
    option_candidate = st.sidebar.selectbox('Which Senatorial Candidate Do You Want To See?', df['Candidate'].unique())
    
    spend_candidate = df[df['Candidate'] == option_candidate].groupby('Candidate')['Transportation and Communication','Labor','Supplies and Logistics', "Pol Ads"].sum()
    
    if spend_candidate.values.sum()== 0:
        no_data = '<p style="font-family:sans-serif; color:Red; font-size: 50px;">NO EXPENSES DATA</p>'
        st.write(no_data, unsafe_allow_html=True)
    
    else:
        
        spend_candidate.replace(0, float("NaN"), inplace=True)
        spend_candidate.dropna(how='all', axis=1, inplace=True)
        
        labels = spend_candidate.columns
        values = [x for x in spend_candidate.values.flatten()]
    
        #Using matplotlib
        pie, ax = plt.subplots(figsize=[10,6])
        labels = labels
        plt.pie(x=values, autopct="%.1f%%", labels=labels, pctdistance=1.1, labeldistance=1.25)
        #patches, texts =plt.pie(x=values, autopct="%.1f%%", labels=labels, pctdistance=0.5)
        plt.title("Expenses Breakdown ", fontsize=14);
        #plt.legend(patches, labels, loc='left center', bbox_to_anchor=(-0.1, 1.), fontsize=8)
        plt.tight_layout()
        #plt.legend(loc='best')

        
        # add a circle at the center to transform it in a donut chart
        my_circle=plt.Circle( (0,0), 0.7, color='white')
        p=plt.gcf()
        p.gca().add_artist(my_circle)
        #plt.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))
        plt.show()
        st.header(f"Spending Breakdown of {option_candidate}")
        st.pyplot(plt)
        
elif my_page == 'Top Spenders':
    st.header(f"Top Spenders Within Range")
    df_ts = df[["Candidate","Total Expenditures Incurred","Win"]].sort_values(by = "Total Expenditures Incurred", ascending=False)
    # Page elements
    ts_top = st.slider(label="Max value",min_value= (100000000 * 0.125), max_value=(100000000 *2.0),step=(100000000 * 0.05))
    ts_bot = st.slider(label="Min value",min_value= (00000000 * 0.125), max_value=(100000000 *.11),step=(750000.00))
    # "Top N Spenders"

    # * Visualizing the graph
    df_ts = df_ts.loc[ ( (df_ts["Total Expenditures Incurred"] <= ts_top ) & (df_ts["Total Expenditures Incurred"] >= ts_bot) ),["Candidate","Total Expenditures Incurred","Win"]].nlargest(15, columns="Total Expenditures Incurred")
    plt.figure(figsize=(8, 8))
    ntop = len(df_ts.index)
    plt.title(f"Top 15 Spenders in the Range")
    ts_bp = sns.barplot(
        x='Total Expenditures Incurred',
        y='Candidate',
        hue='Win',
        data=df_ts, dodge=False
    )
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
    
