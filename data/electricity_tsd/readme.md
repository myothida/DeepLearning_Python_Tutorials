### Electricity Transformer Dataset (ETDataset)

### Electricity Transformer Dataset (ETDataset)

The ETDataset (ETT-small) is available for download from [https://github.com/zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset). It includes data from electricity transformers, such as load and oil temperature, collected over a period from July 2016 to July 2018. The datasets are preprocessed and stored as .csv files.

##### Datasets included:
- **ETT-small**: Data from 2 electricity transformers at 2 stations.

In the ETT-small dataset, each data point is recorded every minute (marked by m), and they were from two regions of a province in China, named ETT-small-m1 and ETT-small-m2, respectively. Each dataset contains:
- 2 years * 365 days * 24 hours * 4 times per hour = 70,080 data points.

Additionally, hourly-level variants are provided for fast development (marked by h), i.e., ETT-small-h1 and ETT-small-h2. Each data point consists of 8 features, including the date of the point, the predictive value "oil temperature", and 6 different types of external power load features.
