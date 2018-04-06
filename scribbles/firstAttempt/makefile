all:
	g++ main.c -fno-rtti -lcuda `llvm-config --ldflags` -o first `llvm-config --libs --system-libs`

run:
	./first