### Electricity Dataset Description

Download URL : https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014

#### Dataset Characteristics
- **Type**: Time-Series  
- **Subject Area**: Computer Science  
- **Associated Tasks**: Regression, Clustering  
- **Feature Type**: Real  

#### Dataset Dimensions
- **Number of Instances**: 370  
- **Number of Features**: 140,256  

#### Dataset Information
The dataset contains electricity consumption data measured every 15 minutes in kilowatts (kW). It is structured such that each column represents a different client. The time-series data spans multiple years, with the following key points:  

###### Data Integrity
- The dataset has no missing values.  

###### Measurement Units
- Values are recorded in kW. To convert to kilowatt-hours (kWh), divide each value by 4.  

###### Client Information
- Some clients were added after 2011, resulting in zeros for earlier periods.  

###### Time Labels
- Time is recorded in Portuguese local time.  
- Each day consists of 96 measurements (24 hours × 4 intervals).  
- Adjustments for daylight saving time:
  - In March (when the time changes and a day has only 23 hours), all measurements between 1:00 AM and 2:00 AM are zero.  
  - In October (when the time changes and a day has 25 hours), measurements between 1:00 AM and 2:00 AM aggregate the consumption of two hours.  

#### Data Formatting
- **File Format**: `.txt` saved in CSV format with a semicolon (`;`) as the delimiter.  
- **First Column**: Contains date and time strings in the format `yyyy-mm-dd hh:mm:ss`.  
- **Other Columns**: Contain float values representing electricity consumption in kilowatts.  

#### Additional Information
- The dataset is clean and ready for use in tasks like regression or clustering.  
- Each column's data is sequential and can be used for modeling long-term consumption patterns.  

This dataset is ideal for analyzing and forecasting electricity usage patterns, particularly for time-series modeling tasks.
