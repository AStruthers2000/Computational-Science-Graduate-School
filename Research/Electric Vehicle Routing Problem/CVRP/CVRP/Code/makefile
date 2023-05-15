CC=g++
CFLAGS=-c -Wall
LDFLAGS=
SOURCES=main.cpp VRP.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=VRP

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.c.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f *.o
