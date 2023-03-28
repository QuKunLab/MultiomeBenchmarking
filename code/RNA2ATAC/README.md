├── dataset                                                       # Sample data folder
├── methods                                                       # Pipelines for ten methods                     
│   ├── RunLIGER.r                     
│   ├── RunLS_Lab.py                       
│   ├── RunMultiVI.py                
│   ├── RunSeurat.r                          
│   └── RunDANCE                           
│       └── data                    
│       └── DANCE.ipynb                     
│       ├── babel.py                 
│       ├── cmae.py                    
│       └── scmogcn.py                         
└── compare                    
      ├── ComputeCMD.ipynb                                 # Compute CMD values
      ├── ComputeROC.ipynb                                 # Compute ROC values
      ├── ComputeDORC.ipynb                                # Compute PCC&CMD values according to DORC-cell matrix 
      └── ComputeAccesson.ipynb                            # Compute PCC&CMD values according to Accesson-cell matrix           
      ├── Accesson_generation.ipynb                        # Generate peak-Accesson correlation
      └── DORC_generation.ipynb                            # Generate DORC information for each dataset                 