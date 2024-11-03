import streamlit as st
from kafka import KafkaConsumer
import json
import pandas as pd
import plotly.express as px
import time
from datetime import datetime

# Set up Kafka consumer
consumer = KafkaConsumer(
    'telematics_data',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='latest',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

# Initialize an empty DataFrame in session state if it doesn't exist
if 'data' not in st.session_state:
    st.session_state['data'] = pd.DataFrame()

# Streamlit app layout
st.title("Real-Time Telematics Data Dashboard")

# Display filters
vehicle_type = st.selectbox("Vehicle Type", ["All", "Sedan", "SUV", "Truck", "Motorcycle"])
age_range = st.slider("Driver Age Range", 18, 70, (18, 70))

# Display the live data in real time
st.subheader("Live Data Stream")
data_placeholder = st.empty()

# Real-time visualizations
st.subheader("Visualizations")
col1, col2 = st.columns(2)

with col1:
    speed_line_chart = st.empty()  # Placeholder for line chart of speed over time
    age_hist = st.empty()  # Placeholder for driver age distribution histogram

with col2:
    vehicle_type_pie = st.empty()  # Placeholder for vehicle type pie chart
    acc_events_bar = st.empty()  # Placeholder for acceleration events by vehicle type

fuel_eff_scatter = st.empty()  # Placeholder for fuel efficiency vs. engine load scatter plot

# Function to handle streaming data
def stream_data():
    for message in consumer:
        # Deserialize the JSON message
        record = message.value
        new_data = pd.DataFrame([record])

        # Append new data to session_state data
        st.session_state['data'] = pd.concat([st.session_state['data'], new_data], ignore_index=True)

        # Apply filters
        filtered_data = st.session_state['data'].copy()
        if vehicle_type != "All":
            filtered_data = filtered_data[filtered_data['Vehicle Type'] == vehicle_type]
        filtered_data = filtered_data[
            (filtered_data['Driver Age'] >= age_range[0]) & 
            (filtered_data['Driver Age'] <= age_range[1])
        ]

        # Display live data
        data_placeholder.dataframe(filtered_data.tail(10))

        # Generate unique keys using timestamp or record count
        # unique_key_speed_line = f"speed_line_{datetime.now().timestamp()}"
        unique_key_age_hist = f"age_hist_{datetime.now().timestamp()}"
        unique_key_vehicle_pie = f"vehicle_pie_{datetime.now().timestamp()}"
        unique_key_acc_bar = f"acc_bar_{datetime.now().timestamp()}"
        unique_key_fuel_scatter = f"fuel_scatter_{datetime.now().timestamp()}"

        # Update visualizations with unique keys
        # # Line Chart of Average Speed Over Time
        # fig_speed_line = px.line(filtered_data, x="Timestamp", y="Average Speed", title="Average Speed Over Time")
        # speed_line_chart.plotly_chart(fig_speed_line, use_container_width=True, key=unique_key_speed_line)

        # Histogram of Driver Ages
        fig_age_hist = px.histogram(filtered_data, x="Driver Age", nbins=15, title="Driver Age Distribution")
        age_hist.plotly_chart(fig_age_hist, use_container_width=True, key=unique_key_age_hist)

        # Pie Chart of Vehicle Types
        fig_vehicle_pie = px.pie(filtered_data, names="Vehicle Type", title="Distribution of Vehicle Types")
        vehicle_type_pie.plotly_chart(fig_vehicle_pie, use_container_width=True, key=unique_key_vehicle_pie)

        # Bar Chart of Acceleration Events by Vehicle Type
        fig_acc_bar = px.bar(filtered_data.groupby("Vehicle Type")["Acceleration Events"].mean().reset_index(),
                             x="Vehicle Type", y="Acceleration Events",
                             title="Average Acceleration Events by Vehicle Type")
        acc_events_bar.plotly_chart(fig_acc_bar, use_container_width=True, key=unique_key_acc_bar)

        # Scatter Plot of Fuel Efficiency vs. Engine Load
        fig_fuel_scatter = px.scatter(filtered_data, x="Engine Load (%)", y="Fuel Efficiency (mpg)", 
                                      title="Fuel Efficiency vs. Engine Load")
        fuel_eff_scatter.plotly_chart(fig_fuel_scatter, use_container_width=True, key=unique_key_fuel_scatter)

        # Adding a slight delay to reduce data processing load
        time.sleep(1)

# Start the data streaming process
stream_data()
