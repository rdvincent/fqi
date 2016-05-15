TARGET = fqi
ifdef DEBUG
CXXFLAGS = -Og -g -Wall
else
CXXFLAGS = -O3 -Wall
endif

HEADERS = *.h
LINK.o = $(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) $(TARGET_ARCH)
OBJECTS = main.o domain.o random.o radfrac.o tass.o

ifdef PROFILE
CXXFLAGS += -pg
endif

all: $(TARGET)

radfrac.o: radfrac.cpp random.h domain.h

tass.o: tass.cpp random.h domain.h

random.o: random.cpp random.h

domain.o: domain.cpp domain.h

main.o: main.cpp $(HEADERS)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

clean:
	rm *.o



