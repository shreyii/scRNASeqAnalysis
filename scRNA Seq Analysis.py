#Naive Peripheral Blood scRNA seq Analysis 
#Import packages 
import sys
import os
import scanpy
import h5py
import scanpy as sc 
import anndata as ad 
import numpy as np
import pandas as pd
import scvi
import scanpy.external as ext
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import leidenalg

warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

h5_folder_path = 'C:/Users/rshreya/Downloads/ScRNA Seq Raw Data Files/Per Blood Peripheral Blood' #Specify file path 

#List all .h5 files in the folder
h5_files_sham = [os.path.join(h5_folder_path, f) for f in os.listdir(h5_folder_path) if f.startswith('Sham')]
h5_files_HF = [os.path.join(h5_folder_path, f) for f in os.listdir(h5_folder_path) if f.startswith('HF')]
h5_files_N = [os.path.join(h5_folder_path, f) for f in os.listdir(h5_folder_path) if f.startswith('N')]

#Read files
def load_and_concatenate(file_paths):
    adata = sc.read_10x_h5(file_paths[0])
    adata.var_names_make_unique()
    for file_path in file_paths[1:]:
        next_adata = sc.read_10x_h5(file_path)
        next_adata.var_names_make_unique()
        adata = adata.concatenate(next_adata)

    return adata

adata_N = load_and_concatenate(h5_files_N)
adata_sham = load_and_concatenate(h5_files_sham)
adata_HF = load_and_concatenate(h5_files_HF)

#Adding condition 
adata_N.obs['condition'] = 'Naive'
adata_sham.obs['condition'] = 'Sham'
adata_HF.obs['condition'] = 'HF'

#Preprocessing - Quality Control 
def preprocessing(merged_adata):
    sc.pp.filter_cells(merged_adata, min_counts=200) #filter cells with fewer than 200 genes
    sc.pp.filter_genes(merged_adata, min_cells=3) #filter genes found in fewer than 3 cells
    merged_adata.var['mt'] = merged_adata.var_names.str.lower().str.startswith('mt-')
    sc.pp.calculate_qc_metrics(merged_adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    merged_adata = merged_adata[merged_adata.obs['pct_counts_mt'] < 5, :] #filter mitochondrial genes less than 5%
    return merged_adata

adata_N = preprocessing(adata_N)
adata_HF = preprocessing(adata_HF)
adata_sham = preprocessing(adata_sham)

#Normalize counts 
def normalize(merged_adata):
    sc.pp.normalize_total(merged_adata, target_sum=1e4)
    sc.pp.log1p(merged_adata)
    return merged_adata

normalize(adata_N)
normalize(adata_sham)
normalize(adata_HF)

#Save raw counts 
def save_raw(merged_adata):
    merged_adata.raw = merged_adata
    return merged_adata

save_raw(adata_N)
save_raw(adata_sham)
save_raw(adata_HF)

#Feature Selection 
def highly_variable(merged_adata):
    sc.pp.highly_variable_genes(merged_adata, min_mean=0.0125, max_mean=3, min_disp=0.5, batch_key = None, flavor = 'seurat_v3') 
    return merged_adata 

highly_variable(adata_N)
highly_variable(adata_sham)
highly_variable(adata_HF)

def filter_highly_variable(merged_adata):
  merged_adata = merged_adata[:, merged_adata.var.highly_variable]
  return merged_adata

adata_N = filter_highly_variable(adata_N)
adata_sham = filter_highly_variable(adata_sham)
adata_HF = filter_highly_variable(adata_HF)

#Scale data 
def scale(merged_adata):
  sc.pp.scale(merged_adata, max_value=10)
  return merged_adata

adata_N = scale(adata_N)
adata_sham = scale(adata_sham)
adata_HF = scale(adata_HF)

#Concat data across conditions
adata_all = adata_N.concatenate(adata_sham, adata_HF)

adata_naive = adata_all[adata_all.obs['condition'] == 'Naive'].copy()
adata_HF = adata_all[adata_all.obs['condition'] == 'HF'].copy()
adata_Sham = adata_all[adata_all.obs['condition'] == 'Sham'].copy()

#Principle Component Analysis 
def pca(merged_adata):
  sc.tl.pca(merged_adata, svd_solver='arpack')
  return merged_adata

pca(adata_all)
pca(adata_naive)

sc.set_figure_params(scanpy=True, dpi=150, dpi_save=300, frameon=True, fontsize=10)   #Publication quality 

#Perform clustering and UMAP - condition wise
plt.rcParams['font.size'] = 6
def leiden_condition_wise(adata, condition_col): 
    sc.pp.neighbors(adata, n_neighbors=100, n_pcs=10) 
    sc.tl.leiden(adata, resolution=0.8, key_added='leiden') #set seed to 0 for reproducibility
    sc.tl.umap(adata, min_dist=1) #set seed to 0 for reproducibility
    if not pd.api.types.is_categorical_dtype(adata.obs[condition_col]):
        adata.obs[condition_col] = adata.obs[condition_col].astype('category')    
    conditions = adata.obs[condition_col].cat.categories
    
    #Condition wise UMAP 
    for condition in conditions:
        condition_adata = adata[adata.obs[condition_col] == condition]
        sc.pl.umap(condition_adata, color=['leiden'], legend_loc='on data', title=f'UMAP - {condition}')
    
    return adata

leiden_condition_wise(adata_all, 'condition')

#Perform clustering and UMAP  
def leiden(merged_adata):
    sc.pp.neighbors(merged_adata, n_neighbors=100, n_pcs=10)
    sc.tl.leiden(merged_adata, resolution=0.8, key_added='leiden')  #set seed to 0 for reproducibility
    sc.tl.umap(merged_adata, min_dist=1)  #set seed to 0 for reproducibility
    sc.pl.umap(merged_adata, color=['leiden'], legend_loc='on data')
    return merged_adata

leiden(adata_all)
leiden(adata_naive)

#Get Ranked Genes using Wilcoxon 
def rank_genes(merged_adata):
  sc.tl.rank_genes_groups(merged_adata, 'leiden', method='wilcoxon')
  return merged_adata

rank_genes(adata_all)
rank_genes(adata_naive)

sc.pl.rank_genes_groups(adata_naive, n_genes=5, sharey=False) #Plot Rank Genes

markers = sc.get.rank_genes_groups_df(adata_naive, None)
markers = markers[(markers.pvals_adj < 0.05) & 
                  (markers.logfoldchanges > .5)]

grouped_top_markers = {}

for group,data in markers.groupby('group'):
    top_names = data['names'][:20]
    grouped_top_markers[group] = top_names.tolist()

#Analyzing top cluster wise genes for annotation 
print(grouped_top_markers)
grouped_top_markers

#Assign Cell Type - Annotation  
cell_type = { #100, 10, 0.8
    "0": "B cell",
    "1": "B cell",
    "2": "CD4+ T cells",
    "3": "B cell",
    "4": "Neutrophils",
    "5": "CD8+ T cells",
    "6": "Erythrocytes",
    "7": "Erythrocytes",
    "8": "Monocytes",
    "9": "Cytotoxic T Cells",
    "10": "Platelets",
    "11": "Natural Killer Cells",
    "12": "B cell",
    "13": "Monocytes",
    "14": "Neutrophils",
    "15": "B cell",
    "16": "Erythrocytes",
    "17": "Erythrocytes",
    "18": "Naive B Cells",
    "19": "Neutrophils",
    "20": "Mast Cells",
    "21": "Regulatory T Cells"
}

adata_all.obs['cell_type'] = adata_all.obs.leiden.map(cell_type) 
sc.pl.umap(adata_all, color = ['cell_type'], legend_loc='on data')
sc.pl.umap(adata_all, color = ['leiden','cell_type'])
sc.pl.umap(adata_all, color = ['cell_type'])

#Subset only monocytes for the interest of this study 
adata_monocytes_naivePB = adata_naive[adata_naive.obs['leiden'].isin(['8', '13'])]

sc.pl.umap(adata_all[adata_all.obs['condition'] == 'Naive'], color=['Siglec1'], color_map ='magma')
adata_naive = adata_all[adata_all.obs['condition'] == 'Naive'].copy()

pca(adata_monocytes_naivePB)
leiden(adata_monocytes_naivePB)

sc.pl.umap(adata_monocytes_naivePB, color=['Siglec1'])

#UMAP showing Siglec1 in Monocyte Population 
from matplotlib.colors import LinearSegmentedColormap
light_gray_black_cmap = LinearSegmentedColormap.from_list("light_gray_black", ["#D3D3D3", "black"])
sc.pl.umap(
    adata_all[adata_all.obs['condition'] == 'Naive'],
    color=['Siglec1'],
    color_map=light_gray_black_cmap
)

#New column for Siglec1 Occcurence 
adata_monocytes_naivePB.obs['CD169 Occurence'] = [
    'CD169+' if x > 0 else 'CD169-'
    for x in adata_monocytes_naivePB.raw[:, 'Siglec1'].X.toarray().flatten()
]

print(adata_monocytes_naivePB.raw.var.index)
print(adata_monocytes_naivePB.obs['CD169 Occurence'].value_counts())

#Reference Markers from Manuscripts 
DCs = ['Zbtb46','Dpp4','Clec9a', 'Clnk','Clec10a','Cd63','Ms4a7','Ccr7','Lamp3','Fscn1', 'Ccl22', 'Il22ra2','Derl3','Gzmb','Ppp1r14a', 'Cd5'] 
Macs = ['Irf7','Ace','Klf4','Klf2','Csf1r','Maf','Mafb','Spi1','Junb','Srgn','S100a4','S100a6','Ccr2','Ccl9','Plbd1']

DCandMacMarkers = DCs + Macs
var_group_labels = ["DC Genes", "Mac Genes"]

var_group_positions = [
    (0, len(DCs) - 1),  #DC markers
    (len(DCs), len(DCandMacMarkers) - 1)  #Mac markers
]

#Plot the heatmap
sc.pl.heatmap(
    adata_monocytes_naivePB,
    var_names=DCandMacMarkers,      
    groupby="CD169 Occurence",          
    dendrogram=True,                    
    var_group_labels=var_group_labels,  
    var_group_positions=var_group_positions  
)

#QC Plots 
qc_metrics = ["n_genes_by_counts", "total_counts", "pct_counts_mt"]
sc.pl.violin(adata_naive, keys=qc_metrics, jitter=0.4, groupby=None, multi_panel=True, ax=ax)

#Marker Gene Plots 
sc.tl.rank_genes_groups(adata_naive, 'cell_type', method='wilcoxon')
sc.pl.rank_genes_groups_dotplot(adata_naive, n_genes=5)
