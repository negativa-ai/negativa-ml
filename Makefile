kernel_detector: src/tracer/kernel_detector.cpp
	mkdir -p build
	nvcc  -lcuda -I /usr/local/cuda/extras/CUPTI/include -L /usr/local/cuda/extras/CUPTI/lib64/ -lcupti -lspdlog  -Xcompiler -fPIC -shared -o build/libkerneldetector.so $^

build: kernel_detector
	cargo build --release --features gpu

install: build
	cargo install --path . --features gpu
	mkdir -p ~/.negativa_ml/lib
	cp build/libkerneldetector.so ~/.negativa_ml/lib/

test:
	cargo test --features gpu

cpu-only-test:
	cargo test --no-default-features

uninstall:
	cargo uninstall negativa-ml
	rm ~/.negativa_ml/lib/libkerneldetector.so
	 