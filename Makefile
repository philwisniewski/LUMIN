CC = mpicc
INCLUDE_PATH = include
CFLAGS = -Wall -g -I $(INCLUDE_PATH)
TARGET = main
SRC = src

all: $(TARGET)

$(TARGET): $(SRC)/main.c $(SRC)/matrix.c
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)/main.c $(SRC)/matrix.c $(SRC)/try.c

.PHONY: clean
clean:
	rm -rf *.o
	rm -f $(TARGET)