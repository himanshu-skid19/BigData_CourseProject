
# **Telematics Data Analysis Project**
#### By Himanshu Singhal, 220150004

This project demonstrates the end-to-end implementation of telematics data analysis for vehicle insurance. It involves real-time data generation, streaming, and visualization through Kafka and Streamlit, as well as machine learning models for risk prediction and clustering.

---

## **Prerequisites**
Ensure the following software/tools are installed:
- **Kafka**: For data streaming.
- **Python**: Version 3.7+.
- **Streamlit**: For real-time dashboard visualization.
- **Apache Spark**: For big data processing and ML modeling.
- Required Python libraries:
  ```bash
  pip install pyspark kafka-python numpy pandas scipy matplotlib seaborn plotly
  ```

## **Steps to Execute the Project**

### **1. Start Zookeeper**
Zookeeper is required to coordinate Kafka brokers.
```bash
$KAFKA_HOME/bin/zookeeper-server-start.sh $KAFKA_HOME/config/zookeeper.properties
```

### **2. Start Kafka Server**
Start the Kafka broker to enable data streaming.
```bash
$KAFKA_HOME/bin/kafka-server-start.sh $KAFKA_HOME/config/server.properties
```

### **3. Run the Streamlit Dashboard**
Launch the Streamlit dashboard for real-time visualization and analytics.
```bash
streamlit run dashboard.py
```

### **4. Start Data Streaming**
Stream the generated telematics data to Kafka.
```bash
python3 stream_data.py
```

### **5. Generate Telematics Data**
Simulate telematics data for the project. This script generates data in real-time and feeds it into the streaming pipeline.
```bash
python3 data_gen.py
```


## **Folder Structure**
```
BigData_CourseProject/
│
├── models/
│   ├── driver_behav_clust_kmeans/
│   └── risk_pred_rf/
│
├── bda_project.ipynb
├── dashboard.py
├── data_gen.py
├── demo_vid_link.txt
├── README.md
├── test.csv
└── train.csv

---

```
## **System Overview**

### **Components**
1. **Data Generation**
   - `data_gen.py` simulates telematics data, including driver behavior, vehicle details, and environmental factors.

2. **Data Streaming**
   - Real-time streaming via Kafka, with `stream_data.py` acting as the producer.

3. **Dashboard Visualization**
   - Streamlit dashboard (`dashboard.py`) for live data visualization and predictions:
     - Risk predictions
     - Driver behavior clustering
     - Claim frequency and severity analysis.

4. **Machine Learning**
   - Risk prediction, claim frequency, and behavior clustering using Spark MLlib.

---

## **Commands Cheat Sheet**
| Command                                 | Description                                       |
|-----------------------------------------|---------------------------------------------------|
| `$KAFKA_HOME/bin/zookeeper-server-start.sh $KAFKA_HOME/config/zookeeper.properties` | Start Zookeeper.                                 |
| `$KAFKA_HOME/bin/kafka-server-start.sh $KAFKA_HOME/config/server.properties`       | Start Kafka server.                              |
| `streamlit run dashboard.py`            | Launch the dashboard.                            |
| `python3 stream_data.py`                | Start streaming data to Kafka.                   |
| `python3 data_gen.py`                   | Generate real-time telematics data for streaming.|

---

## **Project Workflow**
1. **Data Generation:**
   - Simulate telematics data using `data_gen.py`.

2. **Data Streaming:**
   - Stream the generated data to Kafka using `stream_data.py`.

3. **Dashboard:**
   - Use `dashboard.py` to visualize the data and analyze risk predictions, clustering, and claims.

4. **Machine Learning:**
   - Models for risk prediction, claim frequency, claim severity, and driver clustering are pre-trained and integrated into the dashboard.

---


---


