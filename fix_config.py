import yaml
with open('config.yaml', encoding='utf-8', errors='ignore') as f:
    cfg = yaml.safe_load(f)
cfg['training']['epochs'] = 100
with open('config.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(cfg, f, default_flow_style=False)
print('Done - epochs set to 100')