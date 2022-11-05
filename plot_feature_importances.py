import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_json("final1/feature_importances.json", orient="index").reset_index()
df.columns = ["feature", "importance"]
df = df.sort_values("importance", ascending=False)
print(df)

ax = df.plot(kind="bar", x="feature", y="importance")
ax.bar_label(ax.containers[0])

plt.tight_layout()
plt.savefig("final1/feature_importances.png")

plt.show()
