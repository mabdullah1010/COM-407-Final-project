import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# reproducibility
torch.manual_seed(41)
np.random.seed(41)

# read CSV
my_df = pd.read_csv("shooter_combined_output.csv")
print("Loaded rows:", len(my_df))
print("Columns:", list(my_df.columns))

# define which columns are outputs (labels)
label_cols = ['thrust', 'turnLeft', 'turnRight', 'shoot']

# sanity check: ensure columns exist
for c in label_cols:
    if c not in my_df.columns:
        raise ValueError(f"expected label column '{c}' in CSV")

# features = all other columns
feature_cols = [c for c in my_df.columns if c not in label_cols]
print("Feature columns (count):", len(feature_cols))
print(feature_cols)

# prepare X and y
X = my_df[feature_cols].values.astype(float)
y = my_df[label_cols].values.astype(float)  # multi-label (0/1) floats

# scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=41, shuffle=True
)

# convert to tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)   # shape (N,4)
y_test = torch.FloatTensor(y_test)

# model
class Model(nn.Module):
    def __init__(self, in_features, h1=256, h2=128, h3 = 96, h4=64,out_features=4):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, h4)


        self.out = nn.Linear(h4, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        x = self.out(x)  # logits for BCEWithLogitsLoss
        return x

in_features = X_train.shape[1]
model = Model(in_features=in_features, h1=256, h2=128, h3=96,h4=64, out_features=4)
print(model)

# loss and optimizer
criterion = nn.BCEWithLogitsLoss()  # for multi-label binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# training loop
epochs = 1000
losses = []
for epoch in range(epochs):
    model.train()
    logits = model(X_train)              # shape (N,4)
    loss = criterion(logits, y_train)    # expects floats in [0,1]
    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 2 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:3d} - train loss: {loss.item():.6f}")

# save loss plot
plt.plot(range(epochs), losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title('Training Loss')
plt.savefig('training_loss_thruster_5m.png')
plt.close()

# evaluation
model.eval()
with torch.no_grad():
    logits_test = model(X_test)
    probs_test = torch.sigmoid(logits_test)   # convert logits -> probabilities in [0,1]
    preds_test = (probs_test >= 0.5).float()  # threshold at 0.5

    # per-action accuracy
    per_action_acc = (preds_test == y_test).float().mean(dim=0).numpy()  # shape (4,)
    for i, col in enumerate(label_cols):
        print(f"Accuracy for {col}: {per_action_acc[i]*100:.2f}%")

    # exact-match accuracy (all 4 actions correct)
    exact_match = (preds_test == y_test).all(dim=1).float().mean().item()
    print(f"Exact-match accuracy (all 4 correct): {exact_match*100:.2f}%")

    # show a few example predictions
    print("\nExamples (probabilities, prediction, actual):")
    for i in range(min(8, X_test.shape[0])):
        probs = probs_test[i].numpy()
        pred = preds_test[i].numpy().astype(int)
        actual = y_test[i].numpy().astype(int)
        print(f"{i+1:2d}. probs: {np.round(probs,3)} pred: {pred} actual: {actual}")

# simple overall metrics: per-label precision/recall if needed (optional)
# you can add sklearn.metrics if you want more detailed metrics.

# Save model (optional)
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': scaler
}, "model_multi_action_thruster_5m.pth")

print("Done. Loss plot saved to training_loss.png, model to model_multi_action.pth")