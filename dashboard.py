import streamlit as st
from kafka import KafkaConsumer
import json
import pandas as pd
import plotly.express as px
import time
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler

# Initialize Spark session
spark = SparkSession.builder \
    .appName("TelematicsDataDashboard") \
    .getOrCreate()

# Load the pre-trained pipeline model
rf_model_risk = PipelineModel.load("models/risk_pred_rf")

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

# Latest data prediction section
st.subheader("Latest Prediction Highlight")
latest_prediction_placeholder = st.empty()

# Define the function to create the features column before applying the model
def prepare_features_for_risk_model(data):
    feature_columns_risk = ['Total Miles Driven', 'Average Speed', 'Driving during Night (%)', 
                            'Acceleration Events', 'Braking Events']
    assembler = VectorAssembler(inputCols=feature_columns_risk, outputCol='features')
    return assembler.transform(data)


# Placeholder for labeled prediction results
prediction_results_placeholder = st.empty()

# Function to render the latest data as a highlighted HTML row in a scrollable box
# Function to render the latest data as a highlighted HTML row in a scrollable box
def render_highlighted_row(row_data):
    row_html = f"""
    <div style="overflow-x:auto; max-width:100%; padding:5px;">
        <table style="width:100%; border-collapse: collapse; border: 1px solid #1e1e1e;">
            <tr style="background-color: #2c2f33; color: #e0e0e0; font-weight: normal;">
                {''.join([f'<td style="padding: 4px 6px; border: 1px solid #1e1e1e;">{value}</td>' for value in row_data])}
            </tr>
        </table>
    </div>
    """
    latest_prediction_placeholder.markdown(row_html, unsafe_allow_html=True)

# Function to display labeled prediction results
def display_prediction_results(predictions):
    prediction_html = f"""
    <div style="margin-top: 10px; padding: 10px; background-color: #2c2f33; border-radius: 5px;">
        <h4 style="color: #e0e0e0;">Prediction Results</h4>
        <p style="color: #e0e0e0;">Accident Risk Prediction: <strong>{"High" if predictions['risk'] else "Low"}</strong></p>
    </div>
    """
    prediction_results_placeholder.markdown(prediction_html, unsafe_allow_html=True)

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

        # Convert new data into Spark DataFrame
        spark_data = spark.createDataFrame(new_data)

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


        # Prepare features for the model (add 'features' column)
        spark_data_with_features = prepare_features_for_risk_model(spark_data)

        # Use the risk prediction model to get predictions
        risk_prediction = rf_model_risk.transform(spark_data_with_features).select("prediction").collect()[0][0]

        # Add the prediction to the latest data row and highlight it
        latest_row = new_data.iloc[0].to_dict()
        latest_row['Accident Risk Prediction'] = 'High' if risk_prediction else 'Low'
        display_prediction_results({'risk': risk_prediction})

        render_highlighted_row(list(latest_row.values()))

        # Adding a slight delay to reduce data processing load
        time.sleep(0.4)

# Start the data streaming process
stream_data()
