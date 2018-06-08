tests:
	thrust/saxpy
	yagal/saxpy
	cpu/saxpy

build-tests:
	cd thrust && rm -f saxpy && make && cd ..
	cd yagal && rm -f saxpy &&  make && cd ..
	cd cpu && rm -f saxpy && make && cd ..