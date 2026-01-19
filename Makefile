checkpoint_path_arg = --checkpoint_path $(checkpoint_path)
splits_arg = --splits $(splits)
selected_words_to_arg = --selected_words_to $(selected_words_to)
run_python_module = $(work_dir) && python -m

work_dir = cd src/
ifeq ($(OS), Windows_NT)
	local_setup = export LOCAL_DEV=1
	cpu_setup = export USE_CPU=1
else
	local_setup = set LOCAL_DEV=1
	cpu_setup = set USE_CPU=1
endif



parallel_train:
	$(run_python_module) modelling.parallel_train

train:
	$(run_python_module) modelling.train

cpu_train:
	$(cpu_setup) && $(run_python_module) modelling.train

local_train:
	$(local_setup) && $(run_python_module) modelling.train

export_model:
	$(run_python_module) modelling.export $(checkpoint_path_arg)

local_export_model:
	$(local_setup) && $(run_python_module) modelling.export $(checkpoint_path_arg)

onnx_benchmark:
	$(run_python_module) modelling.onnx_benchmark $(checkpoint_path_arg)

local_onnx_benchmark:
	$(local_setup) && $(run_python_module) modelling.onnx_benchmark $(checkpoint_path_arg)

preprocess_data:
	$(run_python_module) data.dataset_preprocessing $(splits_arg) $(selected_words_to_arg)

local_preprocess_data:
	$(local_setup) && $(run_python_module) data.dataset_preprocessing $(splits_arg) $(selected_words_to_arg)

.PHONY: train local_train export_model local_export_model onnx_benchmark local_onnx_benchmark
