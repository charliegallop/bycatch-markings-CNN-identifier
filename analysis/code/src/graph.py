import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/home/charlie/Documents/Uni/Exeter - Data Science/Dissertation/analysis/saved_models/mobilenet_dolphin_losses_100_2048_4.csv")

print(df.head())

fig, ax = plt.subplots(figsize = (10, 10))

ax.plot(df['train_loss'])
ax.plot(df['val_loss'])
plt.show()
