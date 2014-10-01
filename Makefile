TARGET = fqi
CXXFLAGS = -Og -g -Wall
HEADERS = *.h
LINK.o = $(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) $(TARGET_ARCH)
OBJECTS = main.o domain.o random.o

profile: CXXFLAGS += -pg
profile: $(TARGET)

all: $(TARGET)

random.o: random.cpp random.h

domain.o: domain.cpp domain.h

main.o: main.cpp $(HEADERS)

$(TARGET): $(OBJECTS)
	$(CXX) -o $(TARGET) $(OBJECTS)

clean:
	rm *.o



