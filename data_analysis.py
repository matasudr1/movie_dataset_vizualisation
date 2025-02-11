import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

df = pd.read_csv(r"C:\Users\matas\OneDrive\Desktop\movie_project\Top Movies (Cleaned Data).csv")

df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')

today = pd.Timestamp(datetime.now().date())
df = df[df['Release Date'] <= today]

print(df.head())

sns.set(style='whitegrid')

plt.figure(figsize=(10, 6))
sns.histplot(df['Running Time (minutes)'], bins=30, kde=True, color='purple')
plt.title('Distribution of Running Time (minutes)')
plt.xlabel('Running Time (minutes)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
top_genres = df['Genre'].value_counts().head(10)
sns.barplot(x=top_genres.values, y=top_genres.index, palette='viridis')
plt.title('Top 10 Genres by Count')
plt.xlabel('Number of Movies')
plt.ylabel('Genre')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Genre', y='Running Time (minutes)', data=df, palette='pastel')
plt.xticks(rotation=45)
plt.title('Running Time by Genre')
plt.xlabel('Genre')
plt.ylabel('Running Time (minutes)')
plt.show()

df['Release Year'] = df['Release Date'].dt.year

plt.figure(figsize=(12, 6))
yearly_releases = df['Release Year'].value_counts().sort_index()

sns.lineplot(x=yearly_releases.index, y=yearly_releases.values, marker='o')
plt.title('Number of Movies Released Over the Years')
plt.xlabel('Year')
plt.ylabel('Number of Movies')
plt.show()

plt.figure(figsize=(12, 10))
financial_columns = [
    'Production Budget (USD)', 'Domestic Gross (USD)', 'Worldwide Gross (USD)', 
    'Domestic Box Office (USD)', 'International Box Office (USD)', 'Opening Weekend (USD)', 
    'Infl. Adj. Dom. BO (USD)'
]
corr_matrix = df[financial_columns].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap (Financial Metrics)')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='Production Budget (USD)', y='Worldwide Gross (USD)', data=df, 
    hue='Genre', alpha=0.7, palette='muted'
)
plt.title('Production Budget vs Worldwide Gross')
plt.xlabel('Production Budget (USD)')
plt.ylabel('Worldwide Gross (USD)')
plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

plt.figure(figsize=(10, 6))
df['Domestic Share (%)'] = df['Domestic Box Office (USD)'] / df['Worldwide Box Office (USD)'] * 100
sns.histplot(df['Domestic Share (%)'], bins=20, color='green')
plt.title('Distribution of Domestic Box Office Share')
plt.xlabel('Domestic Share (%)')
plt.ylabel('Frequency')
plt.show()
