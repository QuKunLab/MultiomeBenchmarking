# Directory Tree:
                                    
      ├── methods                                                       # Pipelines for seven methods                     
      │   ├── RunLIGER.r                     
      │   ├── RunLS_Lab.py                       
      │   ├── RunMultiVI.py                
      │   ├── RunSeurat.r                          
      │   └── dance-main                           
      │       └── examples                    
      │           └── multi_modality               
      │               └── predict_modality          
      │                   ├── babel.py                 
      │                   ├── cmae.py                    
      │                   └── scmogcn.py                         
      └── compare                    
            ├── ComputeCMD.ipynb                                 # Compute CMD values
            ├── ComputeROC.ipynb                                 # Compute ROC values
            ├── ComputeDORC.ipynb                                # Compute PCC&CMD values according to DORC-cell matrix 
            └── ComputeAccesson.ipynb                            # Compute PCC&CMD values according to Accesson-cell matrix           
            ├── Accesson_generation.ipynb                        # Generate peak-Accesson correlation
            └── DORC_generation.ipynb                            # Generate DORC information for each dataset                 