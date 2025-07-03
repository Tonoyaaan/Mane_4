import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import time

NUMBER_OF_EXAMPLES = 5000 
NUMBER_OF_EPOCHS = 100000
LR = 0.001 



def train_data():
    v0 = random.uniform(5, 25)
    alpha = random.uniform(0.0, np.pi / 2)
    # alpha_radian = np.deg2rad(alpha)
    # alpha_degree = np.rad2deg(alpha_radian)
    g = 9.81
    t = random.uniform(0.0,5)

    h = v0 * np.sin(alpha) * t - 0.5 * g * t ** 2 

    return [v0, alpha, t], [h] #tuple

def generate_data():
    examples = [train_data() for _ in range(NUMBER_OF_EXAMPLES)]
    x,y = zip(*examples) #arancnacnum enq input x-y u autput y-y
    x = torch.tensor(x, dtype=torch.float32) # darcnel torch-in hasaneli tiv
    y = torch.tensor(y, dtype=torch.float32)

    return x, y

def main():

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    start = time.time()

    x, y = generate_data()
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 32), # 3 input, 16 hidden layer
        torch.nn.ReLU(), # activation function, nayum e weight-in u haskanum te inchqan karevor e aynn -  ReLU(max,0)
        torch.nn.Linear(32, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 1),
    )

    loss = torch.nn.MSELoss()  # Mean Squared Error loss function
    optimizer = torch.optim.Adam(model.parameters(), LR)  # Adam optimizer 
    
    num_epochs = NUMBER_OF_EPOCHS

    for epoch in range(num_epochs):
        optimizer.zero_grad() # maqrum e gradientnery
        predictions = model(x) # Forward pass
        current_loss = loss(predictions, y)
        current_loss.backward()  # Backward - het e gnum gradientnery nayum e vat/lav
        optimizer.step()  # Update - vortex vat er noric anel, lav er aj enq anum

        print(f"Epoch: {epoch}, Loss: {current_loss.item()}") 
    
    test = [train_data() for _ in range(100)]
    x_test, y_test = zip(*test)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    with torch.no_grad():  # Gradient-nery 0-acnelu hamar
        predictions = model(x_test).squeeze()  # sarqum enq sovorakan zangvac
    
    pred_y = predictions.tolist()
    true_y = y_test.squeeze()


    plt.scatter(pred_y, true_y) #nax modeliny, apa chishty
    plt.grid()
    plt.plot([min(true_y), max(true_y) ], [min(true_y), max(true_y) ], color='red', linewidth=4)
    plt.show()


    end = time.time()
    print(f"Time:  {end - start}")

if __name__ == "__main__":
    main()
