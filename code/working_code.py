### Check In 1
# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
import numpy as np

data = pd.read_csv(
    r'C:\Users\karin\OneDrive - University of Virginia\Second Year\Comp BME\Module-4-Cancer\data\TRAINING_SET_GSE62944_subsample_log2TPM.csv', index_col=0, header=0)  # can also use larger dataset with more genes
metadata_df = pd.read_csv(
    r'C:\Users\karin\OneDrive - University of Virginia\Second Year\Comp BME\Module-4-Cancer\data\TRAINING_SET_GSE62944_metadata.csv', index_col=0, header=0)

cancer_type = 'COAD'  #Colorectal Cancer
# From metadata, get the rows where "cancer_type" is equal to the specified cancer type
# Then grab the index of this subset (these are the sample IDs)
cancer_samples = metadata_df[metadata_df['cancer_type'] == cancer_type].index
print(cancer_samples)
# Subset the main data to include only these samples
# When you want a subset of columns, you can pass a list of column names to the data frame in []
COAD_data = data[cancer_samples]


desired_gene_list = ['TP53', 'MIR200B', 'KRAS', 'VEGFA' ]
gene_list = [gene for gene in desired_gene_list if gene in COAD_data.index]
for gene in desired_gene_list:
    if gene not in gene_list:
        print(f"Warning: {gene} not found in the dataset.")

# .loc[] is the method to subset by index labels
# .iloc[] will subset by index position (integer location) instead
COAD_gene_data = COAD_data.loc[gene_list]
print(COAD_gene_data.head())

print(COAD_gene_data.describe())
print(COAD_gene_data.var(axis=1))  # Variance of each gene across samples
# Mean expression of each gene across samples
print(COAD_gene_data.mean(axis=1))
# Median expression of each gene across samples
print(COAD_gene_data.median(axis=1))

print(metadata_df.groupby('cancer_type')["gender"].value_counts())

# Explore average age at diagnosis by cancer type
metadata_df['age_at_diagnosis'] = pd.to_numeric(
    metadata_df['age_at_diagnosis'], errors='coerce')
print(metadata_df.groupby(
    'cancer_type')["age_at_diagnosis"].mean())

# Merging datasets
# Merge the subsetted expression data with metadata for COAD samples,
# so rows are samples and columns include gene expression for EGFR and MYC and metadata
COAD_metadata = metadata_df.loc[cancer_samples]
COAD_merged = COAD_gene_data.T.merge(
    COAD_metadata, left_index=True, right_index=True)
print(COAD_merged.head())


# Boxplot of Gene expression in COAD samples using PANDAS directly
COAD_merged[['TP53', 'KRAS', 'VEGFA']].plot.box()
plt.title("Gene Interest Expression in COAD Samples")
plt.show()



plt.figure(figsize=(8, 6))
sns.scatterplot(data=COAD_merged, x='KRAS', y='VEGFA')

plt.title("Crosstalk: KRAS (Evasion) vs. VEGFA (Angiogenesis) in COAD")
plt.xlabel("KRAS Expression Level")
plt.ylabel("VEGFA Expression Level")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

### Check In 2 


#%%
# bring in txt file from class to add genes
gene_list = pd.read_table(r"C:\Users\karin\OneDrive - University of Virginia\Second Year\Comp BME\Module-4-Cancer\data\Menyhart_JPA_CancerHallmarks_core.txt", header=None, index_col = 0)

#print(gene_list.head())
angiogenesis_genes = []
growth_genes = []

angio_gene_pd = gene_list.loc['SUSTAINED ANGIOGENESIS']

for gene in angio_gene_pd:
    angiogenesis_genes.append(gene)

for gene in gene_list.loc['EVADING GROWTH SUPPRESSORS']:
    growth_genes.append(gene)
angio_genes = []
g_genes = []
temp_gene_list = angiogenesis_genes + growth_genes

#print(COAD_merged.head())

#new_COAD_gene_data = COAD_data.loc[full_gene_list]
new_full_gene_list = []
for gene in temp_gene_list:
    if gene in COAD_data.index:
        new_full_gene_list.append(gene)


gene_to_hallmark = {}

for g in angiogenesis_genes:
    gene_to_hallmark[g] = "Angiogenesis"

for g in growth_genes:
    gene_to_hallmark[g] = "Growth Suppression"


#%%
new_COAD_gene_data = COAD_data.loc[new_full_gene_list]
X = new_COAD_gene_data

hallmark_labels = [gene_to_hallmark[g] for g in X.index]
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=hallmark_labels, palette='Set1')
plt.title("PCA of COAD Samples Based on Hallmark Genes")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title='Hallmark')
plt.show()


# %%
dbscan = DBSCAN(eps=0.6, min_samples=5)
y_dbscan = dbscan.fit_predict()
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_dbscan, cmap='viridis')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("DBSCAN Clustering of COAD Samples")
plt.show()

# %%

# First regression - average of all expression levels
x_df = new_COAD_gene_data.loc[[g for g in angiogenesis_genes if g in new_COAD_gene_data.index]].T
x_df = x_df.loc[:,~x_df.columns.duplicated(keep = 'first')]

y_df = new_COAD_gene_data.loc[[g for g in growth_genes if g in new_COAD_gene_data.index]].T
y_df = y_df.loc[:,~y_df.columns.duplicated(keep = 'first')]

    
#print(y_df)

x_val = []
y_val = []
#print(len(y_df.columns))

for i in range(80):
    x_val.append(float(x_df.iloc[i].sum()))
    y_val.append(float(y_df.iloc[i].sum()))


x1 = np.array(x_val).reshape(-1,1)
y1 = np.array(y_val)

min_x = min(x1)
max_x = max(x1)

regression1 = LinearRegression().fit(x1, y1)


x_test = np.linspace(min_x, max_x ,100).reshape(-1,1)
y_test = regression1.predict(x_test)
plt.scatter(x1,y1)
plt.plot(x_test, y_test, color='red')
plt.xlabel("Total Angiogenesis Gene Expression")
plt.ylabel("Total Growth Suppression Gene Expression")
plt.annotate("R^2 = {:.2f}".format(regression1.score(x1,y1)),xy=(0.5,0.9),xycoords='axes fraction',fontsize=14,ha='center')
plt.show()


# %%
# regression 2 - VEGFA vs KRAS

x2_val = []
y2_val = []

for value in x_df['VEGFA']:
    x2_val.append(value)

for value in y_df['KRAS']:
    y2_val.append(value)


x2 = np.array(x2_val).reshape(-1,1)
y2 = np.array(y2_val)

min_x2 = min(x2)
max_x2 = max(x2)

regression2 = LinearRegression().fit(x2,y2)

x_test = np.linspace(min_x2, max_x2 ,100).reshape(-1,1)
y_test = regression2.predict(x_test)
plt.scatter(x2,y2)
plt.plot(x_test, y_test, color='red')
plt.xlabel("VEGFA Expression Level")
plt.ylabel("KRAS Expression Level")
plt.annotate("R^2 = {:.2f}".format(regression2.score(x2,y2)),xy=(0.5,0.9),xycoords='axes fraction',fontsize=14,ha='center')
plt.show()

# %%
# regression 3 - VEGFA vs TP53

x3_val = []
y3_val = []

for value in x_df['VEGFA']:
    x3_val.append(value)

for value in y_df['TP53']:
    y3_val.append(value)


x3 = np.array(x3_val).reshape(-1,1)
y3 = np.array(y3_val)

min_x3 = min(x3)
max_x3 = max(x3)

regression3 = LinearRegression().fit(x3,y3)

x_test = np.linspace(min_x3, max_x3 ,100).reshape(-1,1)
y_test = regression2.predict(x_test)
plt.scatter(x3,y3)
plt.plot(x_test, y_test, color='red')
plt.xlabel("VEGFA Expression Level")
plt.ylabel("KRAS Expression Level")
plt.annotate("R^2 = {:.2f}".format(regression3.score(x3,y3)),xy=(0.5,0.9),xycoords='axes fraction',fontsize=14,ha='center')
plt.show()

# %%

