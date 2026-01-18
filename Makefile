checkpoint_path_arg = --checkpoint_path $(checkpoint_path)
splits_arg = --splits $(splits)
selected_words_to_arg = --selected_words_to $(selected_words_to)

work_dir = cd src/
ifeq ($(OS), Windows_NT)
	local_setup = export LOCAL_DEV=1 && $(work_dir)
	cpu_setup = export USE_CPU=1
else
	local_setup = set LOCAL_DEV=1 && $(work_dir)
	cpu_setup = set USE_CPU=1
endif



train:
	$(work_dir) && python -m modelling.train

cpu_train:
	$(cpu_setup) && python -m modelling.train

local_train:
	$(local_setup) && python -m modelling.train

export_model:
	$(work_dir) && python -m modelling.export $(checkpoint_path_arg)

local_export_model:
	$(local_setup) && python -m modelling.export $(checkpoint_path_arg)

onnx_benchmark:
	$(work_dir) && python -m modelling.onnx_benchmark $(checkpoint_path_arg)

local_onnx_benchmark:
	$(local_setup) && python -m modelling.onnx_benchmark $(checkpoint_path_arg)

preprocess_data:
	python -m data.dataset_preprocessing $(splits_arg) $(selected_words_to_arg)

local_preprocess_data:
	$(local_setup) && python -m data.dataset_preprocessing $(splits_arg) $(selected_words_to_arg)

.PHONY: train local_train export_model local_export_model onnx_benchmark local_onnx_benchmark
