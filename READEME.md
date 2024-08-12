input_folder
load_path
ga_chamfer
check_inter
arap_start_iter: 500
init_type: AG
video_save_dir
render_type: fixed
save_inter: 500

python main_sc4d_arap.py train_dynamic=True input_folder=/mnt/nas_3d_huakun/wzj/synthetic/blooming_rose save_path=./logs/blooming_rose use_arap=True

python main_sc4d_arap_mt.py train_dynamic=True save_path=./logs/flying_ironman prompt="A photo of Batman"