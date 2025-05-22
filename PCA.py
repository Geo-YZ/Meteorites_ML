import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

input_file_path = './Total_Data.xlsx'
output_file_path = './Total_Data_pca_results.xlsx'

df = pd.read_excel(input_file_path, sheet_name='Clean')
data = df.iloc[:, 2:14]  

feature_names = data.columns.tolist()
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
pca = PCA(n_components=10)
data_pca = pca.fit_transform(data_scaled)

weights = pca.components_.T
weights_df = pd.DataFrame(weights, columns=[f'PC{i+1}' for i in range(10)], index=feature_names)
explained_variance_ratio = pca.explained_variance_ratio_
explained_variance_df = pd.DataFrame(explained_variance_ratio, columns=['Explained Variance Ratio'], index=[f'PC{i+1}' for i in range(10)])

eigenvalues = pca.explained_variance_
eigenvalues_df = pd.DataFrame(eigenvalues, columns=['Eigenvalue'], index=[f'PC{i+1}' for i in range(10)])

pca_df = pd.DataFrame(data_pca, columns=[f'PC{i+1}' for i in range(10)])
pca_results = pd.concat([pca_df, weights_df.reset_index()], axis=1)
pca_results.columns = [f'PC{i+1}' for i in range(10)] + ['Variable'] + [f'PC{i+1}_Weight' for i in range(10)]
pca_results = pd.concat([pca_results, explained_variance_df.reset_index(), eigenvalues_df.reset_index()], axis=1)

print("Eigenvalues for each principal component:")
for i, eigenvalue in enumerate(eigenvalues):
    print(f"PC{i+1}: {eigenvalue:.4f}")

pca_results.to_excel(output_file_path, index=False)

print(f"PCA completed '{output_file_path}'")