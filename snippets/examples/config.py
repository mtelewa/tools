import yaml

with open("config.yaml", 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)

for section in cfg:
    print(section)
print(cfg['mysql'])
print(cfg['other'])

