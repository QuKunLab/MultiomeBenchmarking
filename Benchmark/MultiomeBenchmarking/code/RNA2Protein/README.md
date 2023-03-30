# Directory Tree:
    ├── ConvertDataFormat.ipynb                                       # Convert other formats to h5ad format
    ├── dataset                                                       # Sample data folder
    ├── methods                                                       # Pipelines for ten methods 
    │   ├── RuncTP-net.py                                              
    │   ├── RunDengkw.py                       
    │   ├── RunLIGER.r                     
    │   ├── RunscArches.py                       
    │   ├── RunsciPENN.py                 
    │   ├── RunSeurat.r               
    │   ├── RuntotalVI.py                          
    │   └── dance
    │       ├── data                                   # The specific location for data preprocessed by running PrepareData.py 
    │       ├── PrepareData.py                         # Running PrepareData.py to generate preprocessed data for BABEL/CMAE/
    │       ├── babel.py                 
    │       ├── cmae.py                    
    │       └── scmogcn.py                         
    └── compare                                      
        ├── ComputePCC&CMD.ipynb                        # Compute PCC and CMD values
        └── ComputeRC&RU.ipynb                          # Determine RC and RU proteins,then compute PCC and CMD values respectively                      
