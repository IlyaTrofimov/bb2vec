CC=g++
CFLAGS=-O3 -g -Wall -pthread -std=c++0x -l boost_program_options

all: acc-rec

acc-rec: acc-rec.cpp main.cpp global.h
	$(CC) main.cpp acc-rec.cpp -o acc-rec $(CFLAGS)

clean:
	rm -f acc-rec test_runner

test: test_runner

test_runner: *.cpp *.h
	cxxtestgen --error-printer -o test_runner.cpp TestSuite1.h
	$(CC) -o test_runner -Icxxtest test_runner.cpp acc-rec.cpp $(CFLAGS) -Wno-write-strings
	./test_runner;
