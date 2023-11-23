from typing import List
from jsonargparse import ArgumentParser, ActionConfigFile
import yaml
import tempfile

def main():
    # Parse command line arguments
    parser = ArgumentParser(default_config_files=['configs/eval.yaml'])
    parser.add_argument('--configs', type=List[str])
    parser.add_argument('--eval_configs', action=ActionConfigFile)
    args = parser.parse_args()

    combined_app = []

    # Load each YAML file and combine the 'app' sections
    for file in args.configs:
        with open(file, 'r') as f:
            data = yaml.safe_load(f)
        combined_app.extend(data['app'])

    # Create a new dictionary with 'app' as the key and the combined list as the value
    combined_data = {'app': combined_app}

    # # Create a temporary file
    # temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)

    # # Write the combined data to the temporary file
    # yaml.safe_dump(combined_data, temp_file)

    # # Close the file
    # temp_file.close()

    # print(f"Data written to temporary file: {temp_file.name}")
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as f:
        yaml.safe_dump(combined_data, f)
        print(f.name)

if __name__ == '__main__':
    main()