# src/eda.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_demographic_distributions(data, figures_path):
    """Plot distributions of race, sex, and age categories."""
    plt.figure(figsize=(8,6))
    sns.countplot(x='race', data=data)
    plt.title('Race Distribution')
    plt.xlabel('Race')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'race_distribution.png'))
    plt.close()
    
    plt.figure(figsize=(6,4))
    sns.countplot(x='sex', data=data)
    plt.title('Gender Distribution')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'gender_distribution.png'))
    plt.close()
    
    plt.figure(figsize=(10,6))
    sns.countplot(x='age_cat', data=data)
    plt.title('Age Category Distribution')
    plt.xlabel('Age Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'age_category_distribution.png'))
    plt.close()

def plot_decile_score_distribution(data, figures_path):
    """Plot decile score distributions across race, gender, and age category."""
    plt.figure(figsize=(8,6))
    sns.boxplot(x='race', y='decile_score', data=data)
    plt.title('Decile Score by Race')
    plt.xlabel('Race')
    plt.ylabel('Decile Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'decile_score_by_race.png'))
    plt.close()
    
    plt.figure(figsize=(6,4))
    sns.boxplot(x='sex', y='decile_score', data=data)
    plt.title('Decile Score by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Decile Score')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'decile_score_by_gender.png'))
    plt.close()
    
    plt.figure(figsize=(10,6))
    sns.boxplot(x='age_cat', y='decile_score', data=data)
    plt.title('Decile Score by Age Category')
    plt.xlabel('Age Category')
    plt.ylabel('Decile Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'decile_score_by_age_category.png'))
    plt.close()

def plot_recidivism_rates(data, figures_path):
    """Plot actual recidivism rates across race and gender."""
    recid_race = data.groupby('race')['is_recid'].mean().reset_index()
    sns.barplot(x='race', y='is_recid', data=recid_race)
    plt.title('Recidivism Rate by Race')
    plt.xlabel('Race')
    plt.ylabel('Recidivism Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'recidivism_rate_by_race.png'))
    plt.close()
    
    recid_gender = data.groupby('sex')['is_recid'].mean().reset_index()
    sns.barplot(x='sex', y='is_recid', data=recid_gender)
    plt.title('Recidivism Rate by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Recidivism Rate')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'recidivism_rate_by_gender.png'))
    plt.close()

def plot_correlation_heatmap(data, figures_path):
    """Plot correlation heatmap of numerical features."""
    plt.figure(figsize=(12,10))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'correlation_heatmap.png'))
    plt.close()

def perform_eda(processed_data_path, figures_path):
    """Complete EDA pipeline."""
    data = pd.read_csv(processed_data_path)
    plot_demographic_distributions(data, figures_path)
    plot_decile_score_distribution(data, figures_path)
    plot_recidivism_rates(data, figures_path)
    plot_correlation_heatmap(data, figures_path)
    print("EDA completed and figures saved.")

if __name__ == "__main__":
    processed_data_path = 'data/processed/compas_processed.csv'
    figures_path = 'reports/figures'
    os.makedirs(figures_path, exist_ok=True)
    perform_eda(processed_data_path, figures_path)
