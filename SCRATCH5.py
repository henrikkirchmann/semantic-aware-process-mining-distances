import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Recreate the table data
data = {
    "Smoothing method": [
        "Good-Turing", "Witten–Bell", "absolute disc. of 0.9", "original Kneser–Ney", "modified Kneser–Ney",
        "Witten–Bell", "absolute disc. of 0.5", "absolute disc. of 0.8", "absolute disc. of 0.9",
        "absolute disc. of 1.0", "original Kneser–Ney", "modified Kneser–Ney"
    ],
    "Interpolation": ["no", "no", "no", "no", "no", "yes", "yes", "yes", "yes", "yes", "yes", "yes"],
    "PER": [8.28, 8.00, 11.71, 8.62, 9.18, 7.68, 8.03, 7.56, 7.52, 8.24, 6.86, 6.75],
    "std": [0.12, 0.13, 0.14, 0.13, 0.13, 0.12, 0.13, 0.13, 0.12, 0.13, 0.12, 0.12]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Reverse order for correct alignment with table
df = df.iloc[::-1]

# Set figure size
fig, ax = plt.subplots(figsize=(8, 6))

# Create horizontal bar plot
sns.barplot(data=df, x="PER", y="Smoothing method", hue="Interpolation", dodge=False, ax=ax, palette=["gray", "black"])

# Add error bars
for index, row in df.iterrows():
    ax.errorbar(row["PER"], index, xerr=row["std"], fmt='none', color='black', capsize=3)

# Customize labels and limits
ax.set_xlabel("PER ± 1 sd")
ax.set_ylabel("")
ax.set_xlim(6, 12)
plt.legend(title="Interpolation", loc="lower right")

# Remove y-axis labels and ticks to align with table
ax.set_yticks([])
ax.set_yticklabels([])

# Create the table data
table_data = df[["Smoothing method", "Interpolation", "PER"]].copy()
table_data["PER"] = table_data["PER"].astype(str) + " ± " + df["std"].astype(str)  # Format error values

# Create and position the table **inside the plot**
table = plt.table(cellText=table_data.values,
                  colLabels=["Smoothing Method", "Interpolation", "PER ± 1 sd"],
                  cellLoc='center', colLoc='center',
                  loc='center', bbox=[-0.55, 0.25, 1.3, 1])  # Adjust position for integration

# Adjust font size
table.auto_set_font_size(False)
table.set_fontsize(10)

# Final adjustments
plt.tight_layout()
plt.show()
