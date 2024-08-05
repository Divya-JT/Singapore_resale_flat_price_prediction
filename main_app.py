import streamlit as st
from streamlit_option_menu import option_menu
import json
import requests
import pandas as pd
from geopy.distance import geodesic
import statistics
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

import os

cwd = os.getcwd()  # current working directory (cwd)
files = os.listdir(cwd)  # all the files in that directory
#st.write("Files in %r: %s" % (cwd, files))

# lat and long 

data = pd.read_csv("archive/mrt_lrt_data.csv")
mrt_location = pd.DataFrame(data)


with st.sidebar:
    selected = option_menu("Main Menu", ["About Project", "Predictions"])


if selected == "About Project":
    st.markdown("# :blue[Singapore Resale Flat Prices Prediction]")
    st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)
    st.markdown("### :blue[Technologies :] Python, Pandas, Numpy, Scikit-Learn, Streamlit, Python scripting, "
                "Machine Learning, Data Preprocessing, Visualization, EDA, Model Building, Data Wrangling, "
                "Model Deployment")
    st.markdown("### :blue[Overview :] This project aims to construct a machine learning model and implement "
                "it as a user-friendly online application in order to provide accurate predictions about the "
                "resale values of apartments in Singapore. This prediction model will be based on past transactions "
                "involving resale flats, and its goal is to aid both future buyers and sellers in evaluating the "
                "worth of a flat after it has been previously resold. Resale prices are influenced by a wide variety "
                "of criteria, including location, the kind of apartment, the total square footage, and the length "
                "of the lease. The provision of customers with an expected resale price based on these criteria is "
                "one of the ways in which a predictive model may assist in the overcoming of these obstacles.")
    st.markdown("### :blue[Domain :] Real Estate")

if selected == "Predictions":
    st.markdown("# :blue[Predicting Results based on Trained Models]")
    st.markdown("### :orange[Predicting Resale Price (Regression Task) (Accuracy: 87%)]")

    try:
        with st.form("form1"):
            street_name = st.text_input("Street Name")
            block = st.text_input("Block Number")
            floor_area_sqm = st.number_input('Floor Area (Per Square Meter)', min_value=1.0, max_value=500.0)
            lease_commence_date = st.number_input('Lease Commence Date')
            storey_range = st.text_input("Storey Range (Format: 'Value1' TO 'Value2')")

            
            submit_button = st.form_submit_button(label="PREDICT RESALE PRICE")

            if submit_button:
                with open(r"SRFPA/model.pkl", 'rb') as file:
                    loaded_model = pickle.load(file)
                with open(r'SRFPA/scaler.pkl', 'rb') as f:
                    scaler_loaded = pickle.load(f)

                st.write("loaded_model :", loaded_model)
                st.write("scaler_loaded :", scaler_loaded)

                # Calculating lease remain years using lease commence date
                lease_remain_years = 99 - (2023 - lease_commence_date)
                #print(lease_remain_years)
                # Calculating median of storey_range 
                split_list = storey_range.split(' TO ')
                float_list = [float(i) for i in split_list]
                storey_median = statistics.median(float_list)

                # Getting the address 
                address = block + " " + street_name
                query_address = address
                query_string = f'https://www.onemap.gov.sg/api/common/elastic/search?searchVal={query_address}&returnGeom=Y&getAddrDetails=Y'
                resp = requests.get(query_string)

                # -----Using OpenMap API getting the latitude and longitude location of that address-----
                origin = []
                data_geo_location = json.loads(resp.content)
                if data_geo_location['found'] != 0:
                    latitude = data_geo_location['results'][0]['LATITUDE']
                    longitude = data_geo_location['results'][0]['LONGITUDE']
                    origin.append((latitude, longitude))

                # -----Appending the Latitudes and Longitudes of the MRT Stations-----
                # Latitudes and Longitudes are been appended in the form of a tuple  to that list
                mrt_lat = mrt_location['lat']
                mrt_long = mrt_location['lng']
                list_of_mrt_coordinates = []
                for lat, long in zip(mrt_lat, mrt_long):
                    list_of_mrt_coordinates.append((lat, long))

                # -----Getting distance to nearest MRT Stations (Mass Rapid Transit System)-----
                list_of_dist_mrt = []
                for destination in range(0, len(list_of_mrt_coordinates)):
                    list_of_dist_mrt.append(geodesic(origin, list_of_mrt_coordinates[destination]).meters)
                shortest = (min(list_of_dist_mrt))
                min_dist_mrt = shortest
                list_of_dist_mrt.clear()

                # -----Getting distance from CDB (Central Business District)-----
                cbd_dist = geodesic(origin, (1.2830, 103.8513)).meters  # CBD coordinates

                # -----Sending the user enter values for prediction to our model-----
                new_sample = np.array(
                    [[cbd_dist, min_dist_mrt, np.log(floor_area_sqm), lease_remain_years, np.log(storey_median)]])
                new_sample = scaler_loaded.transform(new_sample[:, :5])
                new_pred = loaded_model.predict(new_sample)[0]
                st.write('## :green[Predicted resale price:] ', np.exp(new_pred))

    except Exception as e:
        st.write("Enter the above values to get the predicted resale price of the flat : ", e)


# page config
#  configuration page 
#st.set_page_config(
#    page_title="Singapore Resale Flat Prices Prediction",
#    layout="wide"
#)

