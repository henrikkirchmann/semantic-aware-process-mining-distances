import pandas as pd
import matplotlib.pyplot as plt

# Data Preparation
data = {
    "Model": [
        "Bose 2009 Substitution Scores", "Bose 2009 Substitution Scores", "Bose 2009 Substitution Scores", "Bose 2009 Substitution Scores",
        "De Koninck 2018 act2vec CBOW", "De Koninck 2018 act2vec CBOW", "De Koninck 2018 act2vec CBOW", "De Koninck 2018 act2vec CBOW",
        "Unit Distance", "Unit Distance", "Unit Distance", "Unit Distance",
        "De Koninck 2018 act2vec skip-gram", "De Koninck 2018 act2vec skip-gram", "De Koninck 2018 act2vec skip-gram", "De Koninck 2018 act2vec skip-gram",
        "Activity-Activity Co Occurrence Bag Of Words", "Activity-Activity Co Occurrence Bag Of Words", "Activity-Activity Co Occurrence Bag Of Words", "Activity-Activity Co Occurrence Bag Of Words",
        "Activity-Activity Co Occurrence N-Gram", "Activity-Activity Co Occurrence N-Gram", "Activity-Activity Co Occurrence N-Gram", "Activity-Activity Co Occurrence N-Gram",
        "Activity-Context Bag Of Words", "Activity-Context Bag Of Words", "Activity-Context Bag Of Words", "Activity-Context Bag Of Words",
        "Activity-Context Bag of Words as N-Grams", "Activity-Context Bag of Words as N-Grams", "Activity-Context Bag of Words as N-Grams", "Activity-Context Bag of Words as N-Grams",
        "Activity-Context N-Grams", "Activity-Context N-Grams", "Activity-Context N-Grams", "Activity-Context N-Grams",
        "Activity-Context Bag Of Words PMI", "Activity-Context Bag Of Words PMI", "Activity-Context Bag Of Words PMI", "Activity-Context Bag Of Words PMI",
        "Activity-Context Bag of Words as N-Grams PMI", "Activity-Context Bag of Words as N-Grams PMI", "Activity-Context Bag of Words as N-Grams PMI", "Activity-Context Bag of Words as N-Grams PMI",
        "Activity-Context N-Grams PMI", "Activity-Context N-Grams PMI", "Activity-Context N-Grams PMI", "Activity-Context N-Grams PMI"
    ],
    "Metric": ["Diameter", "Precision@w-1", "Nearest Neighbor", "Triplet Value"] * 12,
    "Value": [
        0.2304, 0.6640, 0.6768, 0.8659,
        0.3177, 0.5197, 0.5641, 0.7586,
        1.0, 0.1111, 0.1205, 0.0,
        0.3177, 0.5197, 0.5633, 0.7584,
        0.7518, 0.1349, 0.00002, 0.6583,
        0.1514, 0.7123, 0.7928, 0.9236,
        1.0, 0.0070, 0.0, 0.0,
        1.0, 0.0070, 0.0, 0.0,
        0.4453, 0.6689, 0.7213, 0.8441,
        1.0, 0.0070, 0.0, 0.0,
        1.0, 0.0070, 0.0, 0.0,
        0.4453, 0.6689, 0.7212, 0.8441
    ]
}

# DataFrame Creation
df = pd.DataFrame(data)

# Visualization
plt.figure(figsize=(14, 8))
for metric in df['Metric'].unique():
    subset = df[df['Metric'] == metric]
    plt.plot(subset['Model'], subset['Value'], marker='o', label=metric)

plt.xticks(rotation=90)
plt.xlabel("Model")
plt.ylabel("Value")
plt.title("Activity Metrics Visualization")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
