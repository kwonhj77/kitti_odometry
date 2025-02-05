import yaml

def load_params(file_name):
    if file_name is None:
        file_name = "default"
    fpath = f'./params/{file_name}.yaml'
    with open(fpath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
