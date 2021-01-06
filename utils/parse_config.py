def parse_model_config(path):
    '''Parses the yolo-v3 layer configuration file and returns module definitions'''
    with open(path, 'r') as file:
        lines = file.read().split('\n')
        # lines = [x for x in lines if x and not x.startswith('#')]
        # lines = [x.strip() for x in lines]
        module_defs = []
        for line in lines:
            line = line.strip()
            if line == '' or line.startswith('#'):
                continue
            if line.startswith('['):
                module_defs.append({})
                module_defs[-1]['type'] = line[1:-1].strip()
                if module_defs[-1]['type'] == 'convolutional':
                    module_defs[-1]['batch_normalize'] = 0
            else:
                key, value = line.split('=')
                module_defs[-1][key.strip()] = value.strip()
    return module_defs


def parse_data_config(path):
    '''Parses the data configuration file'''
    options = {}
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        for line in fp:
            line = line.strip()
            if line == '' or line.startswith('#'):
                continue
            key, value = line.split('=')
            options[key.strip()] = value.strip()
    return options
