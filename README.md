# Single-cell RNA Sequencing Analysis

Author: Shreya Rajasekar 

Citation: https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.124.071772  

This repository contains the scRNA seq analysis run on naive peripheral blood samples from Mus musculus. The analysis was conducted using Scanpy. It includes basic preprocessing, quality control, clustering, and visualization of gene expression patterns following a standard workflow. This study is specifically for analyzing and drawing conclusions on the monocytes subset.

Quality control included filtering cells with fewer than 200 gene counts, removing genes detected in fewer than 3 cells, and excluding cells with over 5% mitochondrial gene content. The data was log-normalized to stabilize variance and to account for differences in sequencing depth. Raw counts were preserved before performing feature selection to identify highly variable genes. Dimensionality reduction was carried out using Principal Component Analysis (PCA), followed by Uniform Manifold Approximation and Projection (UMAP). Leiden algorithm was used to cluster cells with similar transcriptomic profiles. Significant marker genes were identified for each cluster using the Wilcoxon rank-sum test (adjusted p-value < 0.05), and clusters were annotated based on well-established marker genes and databases. Respective violin plots and heatmaps were generated.

References:

1. Wolf, F., Angerer, P. & Theis, F. **SCANPY**: large-scale single-cell gene expression data analysis. *Genome Biol*  **19** , 15 (2018). https://doi.org/10.1186/s13059-017-1382-0 
2. Oscar Franzén, Li-Ming Gan, Johan L M Björkegren,  *PanglaoDB: a web server for exploration of mouse and human single-cell RNA sequencing data* ,  **Database** , Volume 2019, 2019, baz046, [doi:10.1093/database/baz046](https://academic.oup.com/database/article/doi/10.1093/database/baz046/5427041)
3. *Fan-Lin Meng#, Xiao-Ling Huang#, Wen-Yan Qin, Kun-Bang Liu, Yan Wang, Ming Li, Yong-Hong Ren*, Yan-Ze Li* & Yi-Min Sun*. singleCellBase: a high-quality manually curated database of cell markers for single cell annotation across multiple species. Biomarker Research (2023) 11:83.[10.1186/s40364-023-00523-3](https://biomarkerres.biomedcentral.com/articles/10.1186/s40364-023-00523-3)*
4. Congxue Hu, Tengyue Li, Yingqi Xu, Xinxin Zhang, Feng Li, Jing Bai, Jing Chen, Wenqi Jiang, Kaiyue Yang, Qi Ou, Xia Li, Peng Wang, Yunpeng Zhang, CellMarker 2.0: an updated database of manually curated cell markers in human/mouse and web tools based on scRNA-seq data,  *Nucleic Acids Research* , Volume 51, Issue D1, 6 January 2023, Pages D870–D876, [https://doi.org/10.1093/nar/gkac947](https://doi.org/10.1093/nar/gkac947)
