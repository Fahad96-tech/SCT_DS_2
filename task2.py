import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

df = pd.read_csv("sales2_csv.csv", encoding="latin1")

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nMissing Values")
print(df.isnull().sum())

df = df.drop_duplicates()

df = df.fillna(0)

print("\nColumns")
print(df.columns)

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
text_cols = df.select_dtypes(include='object').columns

print("\nSummary Statistics")
print(df[numeric_cols].describe())

value_col = numeric_cols[0]
second_value_col = numeric_cols[1]

category_col = text_cols[0]

top_data = (
    df.groupby(category_col)[value_col]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

sns.barplot(
    x=top_data.index,
    y=top_data.values,
    ax=axes[0, 0]
)

axes[0, 0].set_title("Top Categories")
axes[0, 0].tick_params(axis='x', rotation=30)

sns.histplot(
    df[value_col],
    bins=20,
    kde=True,
    ax=axes[0, 1]
)

axes[0, 1].set_title(f"{value_col} Distribution")

sns.boxplot(
    y=df[value_col],
    ax=axes[1, 0]
)

axes[1, 0].set_title(f"{value_col} Boxplot")

sns.scatterplot(
    x=df[value_col],
    y=df[second_value_col],
    ax=axes[1, 1]
)

axes[1, 1].set_title(f"{value_col} vs {second_value_col}")

plt.tight_layout()

plt.show()
plt.figure(figsize=(10, 6))

sns.heatmap(
    df[numeric_cols].corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()
