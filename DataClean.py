#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


crimes = pd.read_csv("C:/Users/ZEINEB/Downloads/NY_crimes/NYPD_Complaint_Data_Historic.csv")


# In[4]:


crimes.head()


# In[5]:


crimes.describe()


# In[6]:


crimes.info()


# In[7]:


#clean data
crimes = crimes[crimes["Latitude"].notnull()]
crimes = crimes[crimes["Longitude"].notnull()]


# In[8]:


crimes = crimes.drop(['HADEVELOPT', 'HOUSING_PSA', 'PD_CD'], axis = 1)


# In[9]:


crimes = crimes.drop(['CMPLNT_NUM', 'CRM_ATPT_CPTD_CD', 'KY_CD', 'PREM_TYP_DESC', 'Lat_Lon', 'RPT_DT'], axis = 1)


# In[10]:


crimes=crimes.drop(['CMPLNT_TO_DT','CMPLNT_TO_TM','PARKS_NM','JURIS_DESC','JURISDICTION_CODE'], axis = 1)


# In[11]:


crimes=crimes.drop(['SUSP_AGE_GROUP', 'SUSP_RACE', 'SUSP_SEX', 'TRANSIT_DISTRICT' ], axis = 1)


# In[12]:


crimes.info()


# In[13]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import  date
import plotly.express as px #graphic library
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
plt.style.use('ggplot')


# In[14]:


get_ipython().system(' pip install folium')


# In[15]:


import folium
import folium.plugins as plugins
from folium.plugins import HeatMapWithTime
from folium.plugins import HeatMap


# In[16]:


#data exploration 
#map_points function to draw heatmap
def map_points(data_cleaned, lat_col='Latitude', lon_col='Longitude', zoom_start=11,                 plot_points=False, pt_radius=15,                 draw_heatmap=False, heat_map_weights_col=None,                 heat_map_weights_normalize=True, heat_map_radius=15):

    ## center map in the middle of points center in
    middle_lat = data_cleaned[lat_col].median()
    middle_lon = data_cleaned[lon_col].median()

    curr_map = folium.Map(location=[middle_lat, middle_lon],
                          zoom_start=zoom_start)
    # add points to map
    if plot_points:
        for _, row in df.iterrows():
            folium.Marker([row[lat_col], row[lon_col]],
                                radius=pt_radius,
                                popup=row['name'],
                                fill_color="#3db7e4", 
                               ).add_to(curr_map)

    # add heatmap
    if draw_heatmap:
        # convert to (n, 2) or (n, 3) matrix format
        if heat_map_weights_col is None:
            cols_to_pull = [lat_col, lon_col]
        else:
            # if we have to normalize
            if heat_map_weights_normalize:
                data_cleaned[heat_map_weights_col] =                     data_cleaned[heat_map_weights_col] / data_cleaned[heat_map_weights_col].sum()

            cols_to_pull = [lat_col, lon_col, heat_map_weights_col]

        stations = data_cleaned[cols_to_pull]
        curr_map.add_child(plugins.HeatMap(stations, radius=heat_map_radius))

    return curr_map


# In[17]:


#plot heat map
map_points(crimes, lat_col='Latitude', lon_col='Longitude', zoom_start=11, plot_points=False, pt_radius=13, draw_heatmap=True, heat_map_weights_normalize=True, heat_map_radius=13)


# In[18]:


#visualize the number of appearances of different types of crimes 
crimes.OFNS_DESC.value_counts().iloc[:20].plot(kind="barh", title = "Crimes description")


# In[26]:


#visualize the number of crimes per borough
g = sns.scatterplot(x='Longitude', y='Latitude', data=crimes, hue='BORO_NM')
plt.xlabel('Longitude')
plt.ylabel('Latitude)')
g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=4)
plt.show()


# In[20]:


crimes.PATROL_BORO.value_counts().iloc[:20].plot(kind="barh", title = "Location of the crime")


# In[27]:


#Here i want to visualize a graph of distribution representing the number of crimes per  patrol borough
g = sns.scatterplot(x='Longitude', y='Latitude', data=crimes, hue='PATROL_BORO')
plt.xlabel('Longitude')
plt.ylabel('Latitude)')
g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=4)
plt.show()


# In[28]:


#Visualizing the number of crimes per percinct
crimes.ADDR_PCT_CD.value_counts().iloc[:20].plot(kind="barh", title = "crime per percinct")


# 

# In[ ]:





# In[ ]:




