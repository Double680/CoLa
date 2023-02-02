## Instruction of CoLa's training strategy in the ZS-KGC task 

1. Download OpenKE's official code as "OpenKE".
2. Move file "GTransE" into "OpenKE/openke/module/model/" and add init command for GTransE in "OpenKE/openke/module/model/__init__.py".
3. Move file "utils.py", "krl.py" and "train.py" into "OpenKE/".
5. Move data folders "WNV-2K" and "WNV-13K" into "OpenKE/" and rename them as "ImageNet2012" and "ImageNet21K". 
6. Execute "krl.py" first and then "train.py". Remember to modify the related directories and commands.
