import numpy as np
import pandas as pd
import dash
from dash import html
from dash import dcc
import seaborn as sns
import matplotlib.pyplot as plt
#import dash_core_components as dcc
#import dash_html_components as html


from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
default_of_credit_card_clients = fetch_ucirepo(id=350) 
  
# data (as pandas dataframes) 
X = default_of_credit_card_clients.data.features 
y = default_of_credit_card_clients.data.targets 


df_credit = pd.DataFrame(data = X)
df_credit['class']= y

# split dataset by default and not default credit clients
df_credit1 = df_credit[df_credit['class'] == 1]
df_credit0 = df_credit[df_credit['class'] == 0]

# Equal Distribution of classes within Credit dataset
df_credit1_split = df_credit1.head(6000)
df_credit0_split = df_credit0.head(6000)

# Equal Distribution of classes within Credit small dataset
df_credit1_sm = df_credit1.head(150)
df_credit0_sm = df_credit0.head(150)

# merge subsets of data together
df_credit10 = pd.concat( [df_credit1_split,df_credit0_split] ,ignore_index = True)

#small credit subset 
df_credit10_sm = pd.concat( [df_credit1_sm,df_credit0_sm] ,ignore_index = True)

# Divide into test and training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_credit10.iloc[:,:-1].values, df_credit10.iloc[:,-1].values, test_size = 0.2, random_state = 0)

# Apply standard scaling to training and test data to normalize data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)
X_sc = sc.transform(df_credit10.iloc[:,:-1].values)
X_small_cd = sc.transform(df_credit10_sm.iloc[:,:-1].values)

# class variables
y10 = df_credit10.iloc[:,-1].values
y10 = y10.reshape(len(y10),1)
y_s = df_credit10_sm.iloc[:,-1].values
y_s = y_s.reshape(len(y_s),1)

# Defined functions
def sigmoid(x):
    x = np.clip(x, -500, 500)
    result = 1 / (1 + np.exp(-x))
    return result   
    #max_x = np.max(x)
    #result = np.exp(x - max_x) / np.sum(np.exp(x - max_x))
    #return result

def sigmoid_derivative(x):
    smallvalue = 1e-5
    return x * (1 - x) + smallvalue
    #return x * (1 - x) 

def minimize_loss(output,target):
    return np.sum( 1/2 * np.square(target-output) )

def predictions (output):
    predictions = np.zeros_like(output)
    for i in range(len(output)):
        if output[i] > 0.5:
            predictions[i] = 1
        else: 
            predictions[i] = 0
    return predictions

# Neural Network function retreiving optimized weights and biases
def neural_network_fit(X,y1):

    # Initialize weights and biases
    input_neurons = 23
    hidden_neurons = 23
    output_neurons = 1
    
    np.random.seed(42)
    W1 = np.random.uniform(size=(input_neurons, hidden_neurons))
    b1 = np.random.uniform(size=(1, hidden_neurons))
    W2 = np.random.uniform(size=(hidden_neurons, output_neurons))
    b2 = np.random.uniform(size=(1, output_neurons))
    y1 = y1.reshape(len(y1),1)
    
    # Training parameters
    learning_rate = 0.5
    epochs = 15000
    
    # Training loop
    for epoch in range(epochs):
        # Forward propagation
        hidden_input = np.dot(X, W1) + b1
        hidden_output = sigmoid(hidden_input)  
        final_input = np.dot(hidden_output, W2) + b2
        final_output = sigmoid(final_input)
            
        # Calculate error
        error = y1 - final_output
        
        # Calculate loss
        loss = minimize_loss(final_output, y1)
        if epoch % 1000 == 0:        
            print(f"loss is {loss}")
            #print(f"error is {error}")
            
        # Backpropagation
        bp_output = error * sigmoid_derivative(final_output)
        output_bp_grad = bp_output.T.dot(hidden_output)
        error_hidden_layer = bp_output.dot(W2.T)

        bp_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_output)

        
        # Update weights and biases using gradient descent
        W2 += output_bp_grad.T * learning_rate
        b2 += np.sum(bp_output, axis=0, keepdims=True) * learning_rate
        W1 += X.T.dot(bp_hidden_layer) * learning_rate
        b1 += np.sum(bp_hidden_layer, axis=0, keepdims=True) * learning_rate

    print("End of Training:")
    return W1,W2,b1,b2
    #return final_output

# functions for predicting class labels
def neural_network_predict(X,w1,w2,b1,b2):
    # Forward propagation
    h_input = np.dot(X, w1) + b1
    h_output = sigmoid(h_input)   
    f_input = np.dot(h_output, w2) + b2
    f_output = sigmoid(f_input)
    return predictions(f_output)
    
# Training data fit function
W_n1, W_n2, b_n1,b_n2 = neural_network_fit(X_train_sc, y_train)

# fit function on smaller dataset 
W_sm1, W_sm2, b_sm1,b_sm2 = neural_network_fit(X_small_cd, y_s)

# prediction on smaller dataset
y_pred_sm = neural_network_predict(X_small_cd,W_sm1, W_sm2, b_sm1,b_sm2)

# prediction on training set
y_pred_train = neural_network_predict(X_train_sc,W_n1, W_n2, b_n1,b_n2)

# prediction on test results
y_pred_test = neural_network_predict(X_test_sc,W_n1, W_n2, b_n1,b_n2)

## Model Accuracy & Evaluation

# Model Accuracy on smaller dataset
from sklearn.metrics import confusion_matrix, accuracy_score
cm_s = confusion_matrix(y_s,y_pred_sm)
Smaller_dataset_accuracy = accuracy_score(y_s,y_pred_sm)

## accuracy with 23 hidden layer sizes on training data
cm = confusion_matrix(y_train, y_pred_train)
Training_Accuracy = accuracy_score(y_train, y_pred_train)
## acccuracy on test data
Test_Accuracy = accuracy_score(y_test, y_pred_test)

# Measuring accuracy (precision and f1-score) on training dataset
from sklearn.metrics import classification_report
training_class = classification_report(y_train, y_pred_train)


test_class = classification_report(y_test, y_pred_test)

# Calculate residuals
residuals = y_test.reshape(len(y_test),1) - y_pred_test

residuals_unique = np.unique(residuals, return_counts = True)

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Neural Network using Credit Dataset'),
    html.Div(children=f'Accuracy for subset of data with 300 samples: {Smaller_dataset_accuracy * 100:.2f}%'),
    html.Div(children=f'Accuracy for Training data: {Training_Accuracy * 100:.2f}%'),
    html.Div(children='Classification Report for Training Set'),
    html.Pre(children=training_class),
    html.Div(children=f'Accuracy for Test data: {Test_Accuracy * 100:.2f}%'),
    html.Div(children='Classification Report for Test Set'),
    html.Pre(children=test_class),

])

if __name__ == '__main__':
    app.run_server(debug=True)
