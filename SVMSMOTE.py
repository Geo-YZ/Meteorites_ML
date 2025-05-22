import pandas as pd
from imblearn.over_sampling import SVMSMOTE

input_file_path = './Total_Data_pca_results.xlsx'
df = pd.read_excel(input_file_path)

X = df[['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6']]
y = df['group']

svmsmote = SVMSMOTE(random_state=42)
X_resampled, y_resampled = svmsmote.fit_resample(X, y)

resampled_df = pd.DataFrame(X_resampled, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'])
resampled_df['group'] = y_resampled

output_file_path = './Resampled_Data_pca_results.xlsx'
resampled_df.to_excel(output_file_path, index=False)
