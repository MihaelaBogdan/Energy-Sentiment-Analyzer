import os
cfg_path = r'C:\Users\georg\AppData\Local\Programs\Python\Python312\Lib\site-packages\saspy\sascfg_personal.py'
with open(cfg_path, 'r', encoding='utf-8') as f:
    cfg = f.read()

cfg = cfg.replace("'options'  : ['authcode'],", "'options'  : [],")
with open(cfg_path, 'w', encoding='utf-8') as f:
    f.write(cfg)
print('options fixed')
