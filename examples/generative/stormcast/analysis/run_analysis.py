import os
import sys
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.getcwd())
import copy
import json
import subprocess
import shutil
from datetime import datetime
from dateutil.relativedelta import relativedelta
import xarray as xr
from cases_to_nc import save_subset_to_netcdf_fp16
from utils.diagnostics import calculate_divergence_rh
from vertical_section import create_section

def run_addition_inferences(config, registry_file='config/registry.json'):
    additional_config = copy.deepcopy(config)
    initial_time = datetime.strptime(additional_config['initial_time'], '%Y-%m-%d %H:%M')
    new_initial_time = initial_time + relativedelta(months=1)
    additional_config['initial_time'] = new_initial_time.strftime('%Y-%m-%d %H:%M')

    model_shortname = additional_config['model_shortname']
    output_directory = os.path.join(additional_config['output_directory'], "additional")
    os.makedirs(output_directory, exist_ok=True)
    additional_config['output_directory'] = output_directory
    temp_config_file = f'analysis/temp_config_{model_shortname}.json'
    with open(temp_config_file, 'w') as f:
        json.dump(additional_config, f)

    subprocess.run(['python', 'analysis/basic_inference.py', '--config_file', temp_config_file, '--registry_file', registry_file])

    return

def run_inference(config, registry_file='config/registry.json'):
    model_shortname = config['model_shortname']
    with open('analysis/temp_config_{}.json'.format(model_shortname), 'w') as f:
        json.dump(config, f)
    subprocess.run(['python', 'analysis/basic_inference.py', '--config_file', 'analysis/temp_config_{}.json'.format(model_shortname), '--registry_file', registry_file])

def load_case_studies(case_studies_file):
    with open(case_studies_file) as f:
        return json.load(f)

def main(analysis_config, case_studies_file='analysis/case_studies.json', registry_file='config/registry.json'):
    with open(analysis_config) as f:
        base_config = json.load(f)
    
    case_studies = load_case_studies(case_studies_file)

    output_directory = os.path.join(base_config["output_directory"], base_config["model_shortname"])
    os.makedirs(output_directory, exist_ok=True)
    shutil.copy(analysis_config, os.path.join(output_directory, "analysis_config.json"))
    
    for event_name in base_config['event_name']:
        case_study = case_studies[event_name]
        config = base_config.copy()
        config.update(case_study)
        config['output_directory'] = output_directory
        config['event_name'] = event_name
        config['regression_only'] = "False"
        
        run_inference(config, registry_file) 
        run_addition_inferences(config, registry_file)
        config['regression_only'] = "True"
        
        run_inference(config, registry_file)
        run_addition_inferences(config, registry_file)
        event_directory = os.path.join(output_directory, event_name)
        sec_idx = round((config['idx_north'] + config['idx_south']) / 2)
        
        ds_edm = xr.open_dataset(f"{event_directory}/ds_pred_edm.nc")
        ds_noedm = xr.open_dataset(f"{event_directory}/ds_pred_noedm.nc")
        ds_targ = xr.open_dataset(f"{event_directory}/ds_targ.nc")

        create_section(
            ds_targ, 
            ds_edm, 
            ds_noedm, 
            sec_idx,
            base_config["output_directory"]+f"/{event_name}_vertical_section.gif",
            variable='enthalpy',
            config=config,
            )
        
        ds_edm = calculate_divergence_rh(ds_edm)
        ds_noedm = calculate_divergence_rh(ds_noedm)
        ds_targ = calculate_divergence_rh(ds_targ)

        model_name = config['model_shortname']
        os.remove('analysis/temp_config_{}.json'.format(model_name))
        dirname = os.path.join(base_config["output_directory"], (event_name + "__" + model_name + ".nc"))
        save_subset_to_netcdf_fp16(
                           list(range(config["n_steps"])),
                           ds_targ,
                           ds_edm,
                           ds_noedm,
                           config['channel_1'],
                           config['channel_2'],
                           config['channel_3'],
                           config['channel_4'],
                           config['channel_5'],
                           config['channel_2_level'],
                           config['channel_3_level'],
                           config['channel_5_level'],
                           dirname,
                           model_name,
                           event_name,
                           left_lon_idx=config['idx_west'],
                           right_lon_idx=config['idx_east'],
                           bottom_lat_idx=config['idx_north'],
                           top_lat_idx=config['idx_south'],
                    )
    subprocess.run(['python', 'analysis/statistical_analysis.py', output_directory])
    shutil.rmtree(output_directory)
    print("analysis completed")
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run inference on a set of case studies')
    parser.add_argument('--analysis_config', type=str, default='analysis/analysis_config.json', help='Path to the analysis config file')
    parser.add_argument('--registry_file', type=str, default='config/registry.json', help='Path to the registry file')
    parser.add_argument('--case_studies_file', type=str, default='analysis/case_studies.json', help='Path to the case studies file')

    args = parser.parse_args()
    main(args.analysis_config, args.case_studies_file, args.registry_file)

