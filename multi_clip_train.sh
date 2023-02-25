python -m MoCapAct_source.mocapact.distillation.evaluate \
  --policy_path /home/prasaya/cs-projects/MujocoProject/MajorProjectMujoco/multiclip_policy/full_dataset/model/model.ckpt \
  --act_noise 0 \
  --ghost_offset 4 \
  --always_init_at_clip_start \
  --termination_error_threshold 10 \
  --clip_snippets CMU_075_09 \
# other clips are CMU_015_04, CMU_038_03, CMU_049_07, CMU_061_01-172, CMU_069_56, CMU_075_09