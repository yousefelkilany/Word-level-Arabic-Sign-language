SRC_DIR = src
RUN_CMD = cd $(SRC_DIR) && uv run -m

ARGS_CHECKPOINT :=
ARGS_CHECKPOINT += $(if $(checkpoint_path),--checkpoint_path $(checkpoint_path))
ARGS_CHECKPOINT += $(if $(onnx_model_path),--onnx_model_path $(onnx_model_path))

ARGS_TRAIN :=
ARGS_TRAIN += $(if $(selected_signs_from),--selected_signs_from $(selected_signs_from))
ARGS_TRAIN += $(if $(selected_signs_to),--selected_signs_to $(selected_signs_to))
ARGS_TRAIN += $(if $(signers),--signers $(signers))

ARGS_MMAP := $(ARGS_TRAIN)
ARGS_MMAP += $(if $(splits),--splits $(splits))

ARGS_NPZ := $(ARGS_MMAP)
ARGS_NPZ += $(if $(filter 1,$(adjusted)),--adjusted)

ARGS_TRAIN += $(if $(epochs),--epochs $(epochs))

.PHONY: train parallel_train export_model onnx_benchmark visualize_metrics prepare_npz_kps preprocess_mmap_data visualization_dashboard generate_face_map

prepare_npz_kps:
	$(RUN_CMD) data.prepare_npz_kps $(ARGS_NPZ)

preprocess_mmap_data:
	$(RUN_CMD) data.mmap_dataset_preprocessing $(ARGS_MMAP)

train:
	$(RUN_CMD) modelling.train $(ARGS_TRAIN)

parallel_train:
	$(RUN_CMD) modelling.parallel_train $(ARGS_TRAIN)

export_model:
	$(RUN_CMD) modelling.export $(ARGS_CHECKPOINT)

onnx_benchmark:
	$(RUN_CMD) modelling.onnx_benchmark $(ARGS_CHECKPOINT)

visualize_metrics:
	$(RUN_CMD) modelling.visualize_model_performance $(ARGS_CHECKPOINT)

visualization_dashboard:
	$(RUN_CMD) streamlit run modelling/dashboard/app.py

generate_face_map:
	$(RUN_CMD) data.generate_mediapipe_face_symmetry_map

cpu_%: export USE_CPU=1
cpu_%: %
	@:

local_%: export LOCAL_DEV=1
local_%: %
	@:
