# -*- coding: utf-8 -*-
"""
Created on Tue May 27 00:14:43 2025

@author: User
"""

if __name__ == "__main__":
    import os
    import sys
    import torch
    import torch.backends.cudnn as cudnn
    import yaml
    import pandas as pd
    from datetime import datetime
    from train import train
    from utils import AttrDict
    import winsound
    from plyer import notification
    import warnings


    # Setup reproducibility
    cudnn.benchmark = True
    cudnn.deterministic = False
    torch.manual_seed(42)

    def get_config(file_path):
        """Load and validate configuration file"""
        try:
            with open(file_path, 'r', encoding="utf8") as stream:
                opt = yaml.safe_load(stream)
        except Exception as e:
            print(f"Error loading config: {e}")
            sys.exit(1)

        opt = AttrDict(opt)
        
        # Validate required parameters
        required = ['train_data', 'valid_data', 'batch_size', 'num_iter']
        for param in required:
            if param not in opt:
                raise ValueError(f"Missing required parameter: {param}")

        # Character set handling
        if opt.lang_char == 'None':
            chars = set()
            for data in [opt.train_data, opt.valid_data]:
                csv_path = os.path.join(data, 'labels.csv')
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path, sep='^([^,]+),', 
                                      engine='python',
                                      usecols=['filename', 'words'],
                                      keep_default_na=False)
                        chars.update(''.join(df['words']))
                    except Exception as e:
                        print(f"Error reading {csv_path}: {e}")
            opt.character = ''.join(sorted(chars))
        else:
            opt.character = opt.number + opt.symbol + opt.lang_char

        # Add timestamp to experiment name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        opt.original_experiment_name = opt.experiment_name  # Save original name44
        opt.experiment_name = f"{opt.experiment_name}_{timestamp}"
        
        # Create output directory
        os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)
        
        return opt
    
    warnings.filterwarnings("ignore", category=UserWarning, 
                       message="Overriding a previously registered kernel")
    
    # Main training loop
    configs = [
        "config_files/delete/20241025_GRU_ResNet quick test.yaml", # to initialise
        
        # "config_files/20231019_GRU.yaml",
        # "config_files/20231019_LSTM.yaml",
        
        # "config_files/20241025_GRU.yaml",
        # "config_files/20241025_LSTM.yaml",
        
        # "config_files/20240110_GRU.yaml",
        "config_files/20240110_LSTM.yaml"
    ]

    
    for config in configs:               
        print(f"\n{'='*60}")
        print(f"Training {config}")
        print(f"{'='*60}")
        
        try:
            opt = get_config(config)
            with open(f'./saved_models/{opt.experiment_name}/{opt.experiment_name}.yaml', 'w') as f:
                yaml.dump(dict(opt), f)
            
            train(opt, amp=False)
            
            winsound.Beep(700, 500)
            notification.notify(
                title="Training Complete",
                message=f"Finished training {opt.experiment_name}",
                timeout=1000  # duration in seconds
            )

        except Exception as e:
            print(f"Error training {config}: {e}")
            # Error notification with plyer
            winsound.Beep(500, 3000)
            notification.notify(
                title="Training Failed",
                message=f"Error in {config}: {str(e)}",
                timeout=1000
            )
