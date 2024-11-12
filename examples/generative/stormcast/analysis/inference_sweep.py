import json
import os
import shutil
import subprocess


def create_sweep_jsons(sweep_config, sweep_name):
    with open(sweep_config) as f:
        sweep_config = json.load(f)
    
    registry_config_name = sweep_config['registry_config_name']
    sweep_parameter = sweep_config['sweep_parameter']
    sweep_values = sweep_config['sweep_values']
    sweep_tags = sweep_config['sweep_tags']

    with open('config/registry.json') as f:
        registry = json.load(f)

    registry_config = registry["models"][registry_config_name]

    sweep_registry = {"models": {}}

    for sweep_tag, sweep_value in zip(sweep_tags, sweep_values):

        print(f"Creating sweep registry for {sweep_tag} with value {sweep_value}")

        sweep_registry_name = f"{registry_config_name}_sweep_{sweep_name}_{sweep_tag}"

        sweep_registry["models"][sweep_registry_name] = registry_config.copy()

        sweep_registry["models"][sweep_registry_name][sweep_parameter] = sweep_value


    sweep_registry_file = f'analysis/sweep_{sweep_name}_registry.json'
    with open(sweep_registry_file, 'w') as f:
        json.dump(sweep_registry, f, indent=4)
    
    return sweep_registry_file


def submit_sweep(sweep_registry_file, sweep_name, interactive=False, dry_run=False):

    with open(sweep_registry_file) as f:
        sweep_registry = json.load(f)
    
    os.makedirs('analysis/analysis_configs_{}'.format(sweep_name), exist_ok=True)

    for file in os.listdir('analysis/analysis_configs_{}'.format(sweep_name)):
        if file.endswith('.json'):
            os.remove(os.path.join('analysis/analysis_configs_{}'.format(sweep_name), file))

    for key in sweep_registry["models"].keys():

        with open("analysis/analysis_config.json") as f:
            analysis_config = json.load(f)
        
        analysis_config["model_shortname"] = key

        analysis_config_unique_name = f"analysis/analysis_configs_{sweep_name}/analysis_{key}.json"

        with open(analysis_config_unique_name, 'w') as f:

            json.dump(analysis_config, f, indent=4)
        
        cmd = f"python analysis/run_analysis.py --analysis_config {analysis_config_unique_name} --case_studies_file analysis/case_studies.json --registry_file {sweep_registry_file}"

        if dry_run:

            print(cmd)
        
        else:
            if interactive:
                subprocess.run(cmd.split())
            else:
                os.system(f'sbatch batch_inference_sweep.sh "{cmd}"')


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Run inference sweep')
    parser.add_argument('--sweep_config', type=str, help='Path to sweep config file', required=True)
    parser.add_argument('--sweep_name', type=str, help='Name of the sweep', required=True)
    parser.add_argument('--interactive', action='store_true', help='Run the sweep interactively', default=False)
    parser.add_argument('--dry_run', action='store_true', help='Print commands without running them', default=False)

    args = parser.parse_args()

    sweep_registry_file = create_sweep_jsons(args.sweep_config, args.sweep_name)

    submit_sweep(sweep_registry_file, args.sweep_name, interactive=args.interactive, dry_run=args.dry_run)






