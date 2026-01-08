CC = gcc
CFLAGS = -O2 -fPIC -Wno-unused-result#apparently gcc -O3 -march=native -ftree-vectorize simd_example.c -o simd_example will allow vectorisation
LDFLAGS = -shared
TARGET = libfilter.so
SRC = libfilter.c

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

clean:
	rm -f $(TARGET) *.o
