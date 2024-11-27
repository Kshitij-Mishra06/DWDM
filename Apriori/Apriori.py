import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Load the CSV file into a pandas DataFrame
file_path = 'FlipKart_Mobiles.csv'  # Replace with your actual file path
try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print(f"Error: The file at {file_path} was not found.")
    exit()

# Preview the dataset
print("Dataset Preview:")
print(df.head())

# Check if required columns exist in the dataframe
required_columns = ['Model', 'Color', 'Memory', 'Storage']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"Error: Missing columns in the dataset: {missing_columns}")
    exit()

# Convert categorical data into one-hot encoded format for Apriori
df_apriori = pd.get_dummies(df[required_columns])

# Apply the Apriori algorithm
min_support_value = 0.1  # Set the minimum support value (adjust based on your needs)
try:
    frequent_itemsets = apriori(df_apriori, min_support=min_support_value, use_colnames=True)
    print("\nFrequent Itemsets:")
    print(frequent_itemsets)
except ValueError as e:
    print(f"Error in applying Apriori: {e}")
    exit()

# Generate association rules from the frequent itemsets
min_confidence_value = 0.7  # Set the minimum confidence value (adjust based on your needs)
try:
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence_value)
    print("\nAssociation Rules:")
    print(rules)
except ValueError as e:
    print(f"Error in generating association rules: {e}")
    exit()

# Save the results to CSV files
frequent_itemsets.to_csv('frequent_itemsets.csv', index=False)
rules.to_csv('association_rules.csv', index=False)

print("\nFrequent itemsets and association rules have been saved to 'frequent_itemsets.csv' and 'association_rules.csv'.")
