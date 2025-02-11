import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load the dataset (update filename if necessary)
df = pd.read_csv(r"C:\Users\matas\OneDrive\Desktop\movie_project\Top Movies (Cleaned Data).csv")

# Convert 'Release Date' to datetime format
df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')

# Filter out movies with release dates in the future
today = pd.Timestamp(datetime.now().date())
df = df[df['Release Date'] <= today]

# --- Display the first few rows of the cleaned dataset ---
print(df.head())

# Set a consistent style for plots
sns.set(style='whitegrid')

# --- 1. Distribution of Running Time (minutes) ---
plt.figure(figsize=(10, 6))
sns.histplot(df['Running Time (minutes)'], bins=30, kde=True, color='purple')
plt.title('Distribution of Running Time (minutes)')
plt.xlabel('Running Time (minutes)')
plt.ylabel('Frequency')
plt.show()

# --- 2. Top 10 Genres by Count ---
plt.figure(figsize=(10, 6))
top_genres = df['Genre'].value_counts().head(10)
sns.barplot(x=top_genres.values, y=top_genres.index, palette='viridis')
plt.title('Top 10 Genres by Count')
plt.xlabel('Number of Movies')
plt.ylabel('Genre')
plt.show()

# --- 3. Boxplot: Running Time by Genre ---
plt.figure(figsize=(12, 6))
sns.boxplot(x='Genre', y='Running Time (minutes)', data=df, palette='pastel')
plt.xticks(rotation=45)
plt.title('Running Time by Genre')
plt.xlabel('Genre')
plt.ylabel('Running Time (minutes)')
plt.show()

# --- 4. Yearly Trend of Movie Releases ---
# Extract release year after filtering future dates
df['Release Year'] = df['Release Date'].dt.year

plt.figure(figsize=(12, 6))
yearly_releases = df['Release Year'].value_counts().sort_index()

sns.lineplot(x=yearly_releases.index, y=yearly_releases.values, marker='o')
plt.title('Number of Movies Released Over the Years')
plt.xlabel('Year')
plt.ylabel('Number of Movies')
plt.show()

# --- 5. Correlation Heatmap (for relevant financial columns) ---
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

# --- 6. Scatterplot: Production Budget vs Worldwide Gross ---
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

# --- 7. Domestic vs International Box Office Share ---
plt.figure(figsize=(10, 6))
df['Domestic Share (%)'] = df['Domestic Box Office (USD)'] / df['Worldwide Box Office (USD)'] * 100
sns.histplot(df['Domestic Share (%)'], bins=20, color='green')
plt.title('Distribution of Domestic Box Office Share')
plt.xlabel('Domestic Share (%)')
plt.ylabel('Frequency')
plt.show()
