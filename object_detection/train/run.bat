REM 0: train only, 1: train + evaluate, 2: semi-supervised learning, 3: evaluate only
SET mode=2
SET pipeline_config=faster_rcnn_shallownet_dashcol.config

SET tool_path=%~dp0/tool
python %tool_path%/control_main.py --mode=%mode% --work_directory=%wd% --pipeline_config=%pipeline_config%
pause
