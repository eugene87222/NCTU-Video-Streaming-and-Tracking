# -*- coding: utf-8 -*-
import os
import json

if __name__ == '__main__':
    config = json.load(open('config.json', 'r'))
    os.system(f'python -m http.server {config["http_port"]}')
