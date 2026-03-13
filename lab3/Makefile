CXX = g++
CXXFLAGS = -O3 -fopenmp -march=native -I./include
LDFLAGS = -lopenblas -lrt

TARGET = benchmark
SOURCES = syrk.cpp benchmark.cpp
OBJECTS = $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run