SRC_DIR = src
RUN_CMD = cd $(SRC_DIR) && uv run -m

ARGS_CHECKPOINT = --checkpoint_path $(checkpoint_path)
ARGS_DATA       = --splits $(splits) --selected_words_to $(selected_words_to)

.PHONY: train parallel_train export_model onnx_benchmark visualize_metrics preprocess_data visualization_dashboard generate_face_map

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

preprocess_data:
	$(RUN_CMD) data.dataset_preprocessing $(ARGS_DATA)

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
