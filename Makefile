checkpoint_path_arg = --checkpoint_path $(checkpoint_path)
local_setup = export LOCAL_DEV=1

train:
	python train.py

local_train:
	$(local_setup) && python train.py

export_model:
	python export.py $(checkpoint_path_arg)

local_export_model:
	$(local_setup) && python export.py $(checkpoint_path_arg)

onnx_benchmark:
	python onnx_benchmark.py $(checkpoint_path_arg)

local_onnx_benchmark:
	$(local_setup) && python onnx_benchmark.py $(checkpoint_path_arg)

.PHONY: train local_train export_model local_export_model onnx_benchmark local_onnx_benchmark
