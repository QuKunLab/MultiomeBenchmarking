# Directory Tree:
    ├── ConvertDataFormat.ipynb                        # Convert other formats to h5ad format
    ├── dataset # Sample data folder:https://mailustceducn-my.sharepoint.com/:u:/g/personal/hyl2016_mail_ustc_edu_cn/ERRAT_gZ6edCrbvhgMsn6D4BhoJOpH9ALZlA_ZOSw2qzGg?e=8K2fig
    ├── pipeline                                       # Pipelines for eleven methods  
    │   ├── scVAEIT                                        ## Package for running scVAEIT     
    │   ├── RuncTP-net.py                                              
    │   ├── RunDengkw.py                       
    │   ├── RunLIGER.r                     
    │   ├── RunscArches.py                       
    │   ├── RunsciPENN.py
    │   ├── RunscVAEIT.py
    │   ├── RunSeurat.r               
    │   ├── RuntotalVI.py                          
    │   └── dance
    │       ├── data                                       ## The specific location for data preprocessed by running PrepareData.py 
    │       ├── PrepareData.py                             ## Running PrepareData.py to generate preprocessed data for BABEL/CMAE/scMoGNN pipeline 
    │       ├── babel.py                                   ## Use the command "python babel.py --outdir ../dance/ --subtask openproblems_bmmc_cite_phase2_rna --device cuda" to run BABEL algorithm.
    │       ├── cmae.py                                    ## Use the command "python cmae.py --output_path ../dance/ --subtask openproblems_bmmc_cite_phase2_rna --device cuda" to run CMAE algorithm.
    │       └── scmognn.py                                 ## Use the command "python scmogcn.py --subtask openproblems_bmmc_cite_phase2_rna --device cuda" to run scMoGNN algorithm.
    ├── compare                                      
    │   ├── ComputePCC&CMD&RMSE.ipynb                  # Compute PCC, CMD, RMSE values
    │   └── ComputeRC&RU.ipynb                         # Determine RC and RU proteins,then compute PCC, CMD, RMSE values respectively.                     
    │
    └── Results                                        # This file saves the results from each algorithm running on the sample data, which can be used to try the code for metrics calculation. 
