# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
<img width="712" height="860" alt="image" src="https://github.com/user-attachments/assets/d9978ccc-b708-41be-a1a2-8f5dd0d04c23" />

## DESIGN STEPS
### STEP 1: 
Load the customer dataset and preprocess it by handling missing values and encoding categorical features.

### STEP 2: 
Split the dataset into training and testing sets to evaluate model performance.

### STEP 3: 
Define a neural network architecture with fully connected layers and ReLU activation functions.

### STEP 4: 
Select an appropriate loss function (CrossEntropyLoss) and optimizer (Adam) for multi-class classification.

### STEP 5: 
Train the neural network using the training data through forward pass, loss computation, and backpropagation.

### STEP 6: 
Test the trained model on unseen data and predict the customer segment (A, B, C, or D).

## PROGRAM

### Name: HARINI S

### Register Number: 212223240048

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size,32)
        self.fc2 = nn.Linear(32,16)
        self.fc3 = nn.Linear(16,8)
        self.fc4 = nn.Linear(8,4)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
      for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```

### Dataset Information
<img width="1156" height="246" alt="image" src="https://github.com/user-attachments/assets/31b869b1-f7be-4ca9-98d9-87537e23aaa2" />

### OUTPUT
## Confusion Matrix
<img width="725" height="581" alt="image" src="https://github.com/user-attachments/assets/3fbff33e-fc25-427d-b424-6877c21dca48" />

## Classification Report
<img width="570" height="404" alt="image" src="https://github.com/user-attachments/assets/94e79804-ea34-4e3d-b194-1671430869fa" />


### New Sample Data Prediction
<img width="580" height="93" alt="image" src="https://github.com/user-attachments/assets/5cf4c975-aa20-4ce1-92d6-dec061bd8c4c" />

## RESULT
A neural network classification model was successfully developed and trained to accurately predict customer segments (A, B, C, and D) for new market data.
