import os
import scipy

disp_dir = {
    "real_data_SCN_1_scale_2_June032021_OE.mat" : "right2left",
    "real_data_SCN_2_scale_2_June032021_OE.mat" : "right2left",
    "simulated_data_HSDC1_DB_Oct092019_5_OE.mat": "left2right",
    "simulated_data_HSDC2_DB_Oct112019_2_OE.mat": "left2right",
    "simulated_data_HSDC3_DB_Oct112019_1_OE.mat": "left2right",
    "simulated_data_HSDC4_DB_Oct122019_2_OE.mat": "left2right",
}

def load_dataset(path_to_file) -> dict:
    """Loads and validates the format of a dataset adhering to
    our dual camera single disperser CASSI data."""

    # Load dataset if it's a file path
    if isinstance(path_to_file, str):
        if not path_to_file.endswith('.mat'):
            raise ValueError(f'Invalid file format: {path_to_file} must be a .mat file.')
        
        try:
            dataset = scipy.io.loadmat(path_to_file) 
            dataset_name = os.path.split(path_to_file)[1]
            # Augment selected pre-existing datasets with disp_dir
            if dataset_name in disp_dir.keys() and 'disp_dir' not in dataset.keys():
                dataset['disp_dir'] = disp_dir[dataset_name]
        except Exception as e:
            print(f'Check whether file format is compatible with scipy.io or whether file exists.')
            raise
    elif isinstance(path_to_file, dict):
        dataset = path_to_file
    else:
        raise ValueError(f'Dataset must be a .mat file path or dictionary, got {type(path_to_file)}')

    # Validate dataset format and completeness (same for both paths and dicts)
    required_keys = ['Y', 'mask', 'pan_img', 'lambda_calib', 'disp_dir']
    
    errors = []
    
    missing_keys = [key for key in required_keys if key not in dataset.keys()]
    
    # Optional keys
    if 'X' not in dataset.keys():
        print('Warning: spectral image "X" was not found. Reconstruction will operate in real data mode.')
    if 'spectral_sen' not in dataset.keys():
        print('Warning: spectral sensitivy "spectral_sen" of side camera not found. Reconstruction will instantiate it with all ones.')

    if len(missing_keys) > 0:
        errors.append(f'Missing required keys: {missing_keys}')
    
    # Validate shapes and values
    try:
        n1, m2, L = dataset['mask'].shape
        n2 = m2 - L + 1
        
        if dataset['lambda_calib'].size != L:
            errors.append(f'lambda_calib size {dataset["lambda_calib"].size} != mask L dimension {L}')
        # TODO: Consider making the conditional below a strict requirement
        #if dataset['spectral_sen'].size != L:
        #    errors.append(f'spectral_sen size {dataset["spectral_sen"].size} != mask L dimension {L}')
        
        if (dataset['pan_img'].shape[0], dataset['pan_img'].shape[1]) != (n1, n2):
            errors.append(f'pan_img shape {dataset["pan_img"].shape} != expected ({n1}, {n2})')
        
        if dataset['disp_dir'] not in ['left2right', 'right2left']:
            errors.append(f'disp_dir "{dataset["disp_dir"]}" must be "left2right" or "right2left"')
            
    except (KeyError, ValueError, AttributeError) as e:
        errors.append(f'Error validating array shapes: {str(e)}')
    
    if len(errors) > 0:
        raise ValueError(f'Dataset validation failed:\n' + '\n'.join(errors))
    
    return dataset