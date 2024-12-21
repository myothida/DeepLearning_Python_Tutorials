### Project Title: Predicting Car Fuel Consumption Using Neural Networks

#### Project Overview:
In this project, you will build and train a neural network model to predict the fuel consumption (miles per gallon, MPG) of cars based on their features. You will work with the **Auto MPG dataset** from the UCI Machine Learning Repository, which contains information about various car models, including their fuel efficiency. Your task is to create a neural network model using PyTorch (or TensorFlow) that learns to predict the MPG of a car based on its attributes.

#### Objectives:
1. Load and explore the **Auto MPG dataset**. url: https://archive.ics.uci.edu/dataset/9/auto+mpg
2. Preprocess the data (handle missing values, normalize/scale features).
3. Build a neural network model to predict car fuel consumption (MPG).
4. Train and evaluate the model’s performance.
5. Experiment with different architectures and hyperparameters to improve the model's accuracy.

#### Dataset Details:
The dataset contains the following features:
- **mpg**: Fuel efficiency (target variable, miles per gallon)
- **cylinders**: Number of cylinders in the engine
- **displacement**: Engine displacement (in cubic inches)
- **horsepower**: Engine horsepower
- **weight**: Weight of the car (in pounds)
- **acceleration**: Acceleration (time taken to reach 60 mph from a stop)
- **model year**: The year of the car's model
- **origin**: The origin of the car (1 = USA, 2 = Europe, 3 = Japan)

You can access the dataset through the UCI repository, or you can use the version available via Python libraries like `seaborn` or `sklearn`.

#### Helper Files: `data_loader.py` and `data_explorer.py`

To make it easier for you to load the dataset and explore its properties, we have provided two helper files: `data_loader.py` and `data_explorer.py`.

1. **`data_loader.py`**
This file contains a function that loads the Auto MPG dataset for Project 01 and MNIST dataset for Project 02. Here is how to use the helper file. 
```python
# Importing the DataLoader class from data_loader
import data_loader as dl
# Instantiate the DataLoader
data_loader = dl.DataLoader()
# List available datasets
print(data_loader.list_datasets())
# Load the Auto MPG dataset
auto_mpg_data = data_loader.get_dataset("auto_mpg")
```
2. **`data_exploer.py`**
This file allows you to check the basic properties of the data, including basic statistics and visualizations (e.g., box plots). Here’s how to use it:
```Python
import data_explorer as de
data_explorer = de.DataExplorer(auto_mpg_data)
metadata = data_explorer.describe_data()
target = 'mpg'
numerical_cols = auto_mpg_data.select_dtypes(include=[np.number]).columns
feature = numerical_cols[0]
cor_score = data_explorer.check_linear_correlation(feature, target)
outliers = data_explorer.check_outliers(feature)
```

#### Steps to Complete the Project:

1. **Data Loading and Exploration**:
   - Load the dataset and perform an initial exploration to understand the structure of the data.
   - Check for missing values and handle them (either by filling them with the mean or dropping rows).
   - Explore the distribution of features and the target variable (MPG).

2. **Data Preprocessing**:
   - Normalize the features to ensure that they are on a similar scale.
   - Encode categorical variables (e.g., `origin`) as needed.
   - Split the dataset into training and testing sets.

3. **Model Building**:
   - Build a neural network with at least one hidden layer. You should experiment with the number of neurons in the hidden layer(s) and the activation function(s).
   - The network should take in the car’s features (engine size, weight, etc.) and output a predicted MPG.

4. **Model Training**:
   - Train the model using Mean Squared Error (MSE) as the loss function.
   - Use Stochastic Gradient Descent (SGD) or another optimizer of your choice.
   - Track the loss and accuracy of the model over epochs.

5. **Evaluation**:
   - Evaluate the trained model on the test set.
   - Compute the model’s performance using metrics : Mean Squared Error (MSE).
   - Visualize the predicted vs. actual MPG values.

6. **Experimentation**:
   - Try different architectures (e.g., varying the number of hidden layers or neurons).
   - Experiment with different activation functions (ReLU, Sigmoid, Tanh).
   - Test different optimizers (SGD, Adam, etc.), numpber of epoches and learning rates.

7. **Reporting**:
   - Summarize your findings and results.
   - Compare the performance of your model with a simple baseline model (e.g., predicting the mean MPG for all cars).
   - Discuss the results and how your model could be further improved.

#### Deliverables:
- Python code for the entire project (including data loading, preprocessing, model creation, training, evaluation, and experimentation) in the **template notebook file** (*.ipynb).
- A **2-page** report listing your model’s performance against the experiments you performed, and conclusions.
- Graphs/plots comparing predicted vs. actual MPG, as well as loss/accuracy curves during training.

#### Skills You Will Learn:
- Data preprocessing and exploration in Python using libraries like `pandas` and `seaborn`.
- Building, training, and evaluating neural network models using PyTorch (or TensorFlow).
- Experimenting with different neural network architectures and hyperparameters.
- Analyzing model performance and improving accuracy.

#### Suggested Tools and Libraries:
- **PyTorch** or **TensorFlow** for building and training the neural network.
- **pandas**, **matplotlib**, and **seaborn** for data exploration and visualization.
- **scikit-learn** for data preprocessing (e.g., train-test split, normalization).

#### Grading Criteria:
- **Code functionality**: Correct implementation of the data preprocessing, model building, and training.
- **Model performance**: Evaluation of model accuracy and experimentation results.
- **Report quality**: Clear and concise discussion of the methodology, experiments, and findings.
- **Creativity**: Exploring different architectures and improvements to the model.

*Good luck, and feel free to ask for help if needed!*
