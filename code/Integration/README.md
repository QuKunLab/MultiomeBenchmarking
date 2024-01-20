# Directory Tree:
    ├── dataset       # Sample data folder: Please download from https://mailustceducn-my.sharepoint.com/:f:/g/personal/hyl2016_mail_ustc_edu_cn/Eg3p7u9mIsFAhGrjSv_Xm0MBBzm3XklJ-e-3iJzSALeQ7w?e=jBB0RX
    │   ├── Vertical
    │   │   ├── RNA_ATAC                                  ## Sample data for vertical RNA+ATAC integration algorithms test,
    │   │   │                                             ## which includes the RNA, ATAC data,and the celltype information in the meta_data.csv file to compute metrics.                  
    │   │   └── RNA_ADT                                   ## Sample data for vertical RNA+Protein integration algorithms test,
    │   │                                                 ## which includes the RNA, ADT data, and the celltype information in the metadata.csv file to compute metrics. 
    │   ├── Horizontal
    │   │   ├── RNA_ATAC                                  ## Dataset1 for horizontal RNA+ATAC integration algorithms test,
    │   │   │                                             ## which includes the RNA and ATAC data of batch1 and batch2,and the celltype data to compute metrics.                  
    │   │   └── RNA_Protein                               ## Dataset7 for horizontal RNA+Protein integration algorithms test,
    │   │                                                 ## which includes the RNA and ADT data of batch1 and batch2,and the celltype data to compute metrics. 
    │   └── Mosaic
    │       ├── RNA_RNAProtein                            ## Batch1 includes RNA and ADT data, Batch2 includes RNA data for RNA & RNA+Protein mosaic integration algorithms test, 
    │       │                                             ## and the celltype data to compute metrics.            
    │       ├── RNA_RNAATAC                               ## Batch1 includes RNA and ATAC data, Batch2 includes RNA data for RNA & RNA+ATAC mosaic integration algorithms test, 
    │       │                                             ## and the celltype data to compute metrics.   
    │       ├── ATAC_RNAATAC                              ## Batch1 includes RNA and ATAC data, Batch2 includes ATAC data for ATAC & RNA+ATAC mosaic integration algorithms test, 
    │       │                                             ## and the celltype data to compute metrics.   
    │       └── RNAProtein_RNAATAC                        ## Batch1 includes RNA and ATAC data, Batch2 includes RNA and ADT data for RNA+Protein & RNA+ATAC mosaic integration algorithms test, 
    │                                                     ## and the celltype data to compute metrics.   
    │
    ├── pipeline                                       # Pipelines for each algorithm
    │   ├── scVAEIT                                       ## Package for running scVAEIT.    
    │   ├── Vertical
    │   │   ├── RNA_ATAC                                  ## This directory includes twelve vertical integration algorithms for RNA+ATAC senario.
    │   │   │   ├── MIRA_run.py                           ## It is suggested to use "python XX.py path DatasetX" or "Rscript XX.R path DatasetX" commend,              
    │   │   │   ├── MOFA_run.py                           ## in which "path" is the path to RNA and ATAC data, and "DatasetX" is the DataName.
    │   │   │   ├── MOJITOO_run.R                  
    │   │   │   ├── Multigrate_run.py  
    │   │   │   ├── MultiVI_run.py  
    │   │   │   ├── scAI_run.R 
    │   │   │   ├── Schema_run.py  
    │   │   │   ├── scMVP_run.py  
    │   │   │   ├── SCOIT_run.py  
    │   │   │   ├── scVAEIT_run.py
    │   │   │   ├── Seurat_run.R 
    │   │   │   ├── DeepMaps_run_human.R                      ### It should be noted that the DeepMaps algorithm need to be used according to species "human" or "mouse". 
    │   │   │   ├── DeepMaps_run_mouse.R  
    │   │   │   └── filt_DEgene.py                            ### This python file is used to pick top-4000 HVGs and top-30000 peaks.
    │   │   │
    │   │   └── RNA_ADT                                   ## This directory includes nine vertical integration algorithms for RNA+Protein senario.
    │   │       ├── CiteFuse_run.R                        ## It is suggested to use "python XX.py path DatasetX" or "Rscript XX.R path DatasetX" commend,              
    │   │       ├── DeepMaps_run.R                        ## in which "path" is the path to RNA and ADT data, and "DatasetX" is the DataName.
    │   │       ├── MOJITOO_run.R                  
    │   │       ├── Multigrate_run.py  
    │   │       ├── scArches_run.py  
    │   │       ├── SCOIT_run.py 
    │   │       ├── scVAEIT_run.py
    │   │       ├── Seurat_run.R                
    │   │       ├── TotalVI_run.py  
    │   │       └── filt_DEgene.py                            ### This python file is used to pick top-4000 HVGs and top-30000 peaks.    
    │   │           
    │   ├── Horizontal
    │   │   ├── RNA_ATAC                                  ## This directory includes seven horizontal integration algorithms for RNA+ATAC senario.
    │   │   │   ├── RunMIRA.py                            ## It is suggested to change the data path in the .py file and use "python RunXX.py" or "Rscript RunXX.r" commend.              
    │   │   │   ├── RunMOFA.py  
    │   │   │   ├── RunMultigrate.py  
    │   │   │   ├── RunMultiVI.py  
    │   │   │   ├── RunscMoMaT.py  
    │   │   │   ├── RunscVAEIT.py  
    │   │   │   └── RunUINMF.r   
    │   │   └── RNA_Protein                               ## This directory includes five horizontal integration algorithms for RNA+Protein senario.
    │   │       ├── RunMultigrate.py                      ## It is suggested to change the data path in the .py file and use "python RunXX.py" or "Rscript RunXX.r" commend.
    │   │       ├── RunscArches.py  
    │   │       ├── RunscVAEIT.py  
    │   │       ├── RuntotalVI.py  
    │   │       └── RunUINMF.r   
    │   │                                                   
    │   └── Mosaic
    │       ├── RNA_RNAProtein                            ## This directory includes seven mosaic integration algorithms for RNA & RNA+Protein subcase.
    │       │   ├── Multigrate_Mosaic.py                  ## It is suggested to use "python XX.py path1 path2 DatasetX" or "Rscript XX.r path1 path2 DatasetX" commend,        
    │       │   ├── scArches_Mosaic.py                    ## in which "path1" is the path to RNA+Protein data,"path2" is the path to RNA data, and "DatasetX" is the DataName.
    │       │   ├── scMoMaT_Mosaic.py  
    │       │   ├── scVAEIT_Mosaic.py  
    │       │   ├── StabMap_Mosaic.r  
    │       │   ├── totalVI_Mosaic.py  
    │       │   └── UINMF_Mosaic.R                                                       
    │       ├── RNA_RNAATAC                               ## This directory includes six mosaic integration algorithms for RNA & RNA+ATAC subcase. 
    │       │   ├── Multigrate_Mosaic.py                  ## It is suggested to use "python XX.py path1 path2 DatasetX" or "Rscript XX.r path1 path2 DatasetX" commend,
    │       │   ├── MultiVI_Mosaic.py                     ## in which "path1" is the path to RNA+ATAC data,"path2" is the path to RNA data, and "DatasetX" is the DataName.
    │       │   ├── scMoMaT_Mosaic.py  
    │       │   ├── scVAEIT_Mosaic.py 
    │       │   ├── StabMap_Mosaic.r
    │       │   └── UINMF_Mosaic.R                                                    
    │       ├── ATAC_RNAATAC                              ## This directory includes six mosaic integration algorithms for ATAC & RNA+ATAC subcase. 
    │       │   ├── Multigrate_Mosaic.py                  ## It is suggested to use "python XX.py path1 path2 DatasetX" or "Rscript XX.r path1 path2 DatasetX" commend,
    │       │   ├── MultiVI_Mosaic.py                     ## in which "path1" is the path to RNA+ATAC data,"path2" is the path to ATAC data, and "DatasetX" is the DataName.
    │       │   │
    │       │   ├── calc_pseudo_count.R                       ### This R file is used to make GxR.csv for scMoMaT algorithm,
    │       │   ├── scMoMaT_Mosaic_Human.py                        and it should be noted that both this preprocessing file and the scMoMaT algorithm need to be used according to species "Human" or "Mouse". 
    │       │   ├── scMoMaT_Mosaic_Mouse.py  
    │       │   ├── scVAEIT_Mosaic.py 
    │       │   ├── StabMap_Mosaic.r
    │       │   └── UINMF_Mosaic.R                                               
    │       └── RNAProtein_RNAATAC                        ## This directory includes five mosaic integration algorithms for RNA+Protein & RNA+ATAC subcase.
    │           ├── Multigrate_Mosaic.py                  ## It is suggested to use "python XX.py path1 path2 DatasetX" or "Rscript XX.r path1 path2 DatasetX" commend,
    │           ├── scMoMaT_Mosaic.py                     ## in which "path1" is the path to RNA+ATAC data,"path2" is the path to RNA+Protein data, and "DatasetX" is the DataName.
    │           ├── scVAEIT_Mosaic.py 
    │           ├── StabMap_Mosaic.r
    │           └── UINMF_Mosaic.R                                                           
    │
    ├── compare  
    │   ├── count_metrics_ADT.ipynb                    # This .ipynb file is used to compute all metrics for vertical senario of RNA+Protein.
    │   ├── count_metrics_ATAC.ipynb                   # This .ipynb file is used to compute all metrics for vertical senario of RNA+ATAC.
    │   └── ComputeMetrics.py                          # This python file is used to compute all metrics for horizontal senario and mosaic senario. 
    │
    └── results                                        # This file saves the results from each algorithm running on the sample data, which can be used to try the code for metrics calculation.  







        
        