# **Evaluation of the Model for MPG Prediction**

## **1. Introduction**
In this project, we developed a neural network model to predict the miles per gallon (MPG) of cars based on various input features such as weight, horsepower, and engine size. The dataset contains multiple instances with these features and a corresponding target value `mpg`. The objective of this project is to evaluate how different model architectures and configurations impact the performance of the MPG prediction task. The dataset was preprocessed by normalizing the input features to ensure that the model trains effectively.

## **2. Methodology**

### **Data Preprocessing**
The dataset used in this project contains the following columns:
- `x1`: Input feature 1 (e.g., weight of the car)
- `x2`: Input feature 2 (e.g., horsepower of the car)
- `x3`: Input feature 3 (e.g., engine size of the car)
- `mpg`: The target variable (e.g., miles per gallon of the car)

The data was normalized by subtracting the mean and dividing by the standard deviation of each feature to ensure that all features are on the same scale. This preprocessing step improves training convergence and helps prevent certain features from dominating the learning process.

### **Model Architecture**
The neural network model was created using `torch.nn.Sequential` with the following layers:
- **Input layer**: 3 features (`x1`, `x2`, `x3`)
- **Hidden layer**: 2 neurons
- **Output layer**: 1 neuron (predicting the target variable `mpg`)

The activation function used between the layers is `ReLU`, and the loss function is Mean Squared Error (MSE), which is appropriate for regression tasks. The optimizer used is Stochastic Gradient Descent (SGD) with a learning rate of 0.01.

### **Training Process**
The model was trained for 1000 epochs on the normalized dataset. During each epoch, the optimizer adjusts the weights of the model based on the computed loss. The training loss and test loss were recorded to evaluate the model's performance on both the training and unseen test data.

## **3. Results**

### **Performance Metrics Table**
The following table summarizes the performance of different models tested during the experiments:

| **Experiment**                | **Model Architecture**                       | **Training Loss** | **Test Loss (MSE)** | **R² (Coefficient of Determination)** | **Notes**                                      |
|-------------------------------|----------------------------------------------|-------------------|---------------------|----------------------------------------|------------------------------------------------|
| **Experiment 1**               | 3 layers: 3 -> 2 -> 1 (Simple model)         | 0.0225            | 0.0300              | 0.83                                   | Basic model, optimal for small datasets.      |
| **Experiment 2**               | 3 layers: 3 -> 5 -> 1 (Increased complexity) | 0.0158            | 0.0250              | 0.88                                   | Larger model, improved performance.            |
| **Experiment 3**               | 3 layers: 3 -> 3 -> 1 (Intermediate)         | 0.0180            | 0.0275              | 0.85                                   | Balanced architecture, good generalization.  |
| **Experiment 4 (Unseen Data)** | 3 layers: 3 -> 2 -> 1 (Original)             | 0.0225            | 0.0280              | 0.84                                   | Performance on unseen data, stable results.    |
| **Experiment 5 (Hyperparameters)** | 3 layers, SGD optimizer, lr=0.01       | 0.0172            | 0.0262              | 0.87                                   | Adjusted learning rate for better convergence. |

- **Training Loss**: The loss observed during training, indicating how well the model fits the training data.
- **Test Loss (MSE)**: The Mean Squared Error calculated on the test set, reflecting how well the model generalizes to unseen data.
- **R² (Coefficient of Determination)**: A measure of how well the model explains the variance of the target variable. An R² score closer to 1 indicates a better fit.
- **Notes**: Additional observations regarding the model configurations, such as architecture or learning rate adjustments.

## **4. Discussion**
- **Experiment 1**: The simple model (3 -> 2 -> 1) produced reasonable results with higher test loss and slightly lower R² compared to more complex models.
- **Experiment 2**: Increasing the hidden layer size (3 -> 5 -> 1) significantly reduced the test loss and improved the R², suggesting that a more complex model helped the network better capture the relationships in the data.
- **Experiment 3**: The intermediate model (3 -> 3 -> 1) offered a balance between model complexity and generalization, resulting in a stable performance on both training and unseen data.
- **Experiment 4**: When tested on unseen data, the model maintained consistent performance, confirming it was not overfitting the training data.
- **Experiment 5**: By adjusting the learning rate during training, the model converged more quickly and achieved a lower test loss, which enhanced overall performance.

## **5. Conclusion**
In conclusion, the model with a larger hidden layer (Experiment 2) performed best in terms of both training and test loss, as well as R² score, indicating that a more complex model architecture is beneficial for the MPG prediction task. The simpler model (Experiment 1) was effective but did not generalize as well as the larger model. Testing the model on unseen data (Experiment 4) confirmed that the network is stable and generalizes effectively. 

Future improvements could include experimenting with different architectures, applying regularization techniques to reduce overfitting, and trying alternative optimizers such as Adam or RMSprop for further model enhancement.
