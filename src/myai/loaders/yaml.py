from collections.abc import Mapping

import ruamel.yaml

def yamlread(file: str):
    yaml = ruamel.yaml.YAML(typ='safe')
    with open(file, 'r', encoding = 'utf8') as f:
        return yaml.load(f)

def yamlwrite(d: Mapping, file:str, sequence = 4, offset = 2):
    yaml = ruamel.yaml.YAML()
    yaml.indent(sequence=sequence, offset=offset)
    with open(file, 'w', encoding = 'utf8') as f:
        yaml.dump(d, f, )

