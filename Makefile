SRC_DIR = src
RUN_CMD = cd $(SRC_DIR) && uv run -m

ARGS_CHECKPOINT :=
ARGS_CHECKPOINT += $(if $(checkpoint_path),--checkpoint_path $(checkpoint_path))
ARGS_CHECKPOINT += $(if $(onnx_model_path),--onnx_model_path $(onnx_model_path))

ARGS_DATA :=
ARGS_DATA += $(if $(splits),--splits $(splits))
ARGS_DATA += $(if $(signers),--signers $(signers))
ARGS_DATA += $(if $(selected_words_from),--selected_words_from $(selected_words_from))
ARGS_DATA += $(if $(selected_words_to),--selected_words_to $(selected_words_to))

ARGS_NPZ := $(ARGS_DATA)
ARGS_NPZ += $(if $(adjusted),--adjusted $(adjusted))


.PHONY: train parallel_train export_model onnx_benchmark visualize_metrics preprocess_mmap_data visualization_dashboard generate_face_map prepare_npz_kps

train:
	$(RUN_CMD) modelling.train

parallel_train:
	$(RUN_CMD) modelling.parallel_train

export_model:
	$(RUN_CMD) modelling.export $(ARGS_CHECKPOINT)

onnx_benchmark:
	$(RUN_CMD) modelling.onnx_benchmark $(ARGS_CHECKPOINT)

visualize_metrics:
	$(RUN_CMD) modelling.visualize_model_performance $(ARGS_CHECKPOINT)

preprocess_mmap_data:
	$(RUN_CMD) data.mmap_dataset_preprocessing $(ARGS_DATA)

prepare_npz_kps:
	$(RUN_CMD) data.prepare_npz_kps $(ARGS_NPZ)

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
