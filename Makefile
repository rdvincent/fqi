CXXFLAGS = -O3 -g
HEADERS = *.h
LINK.o = $(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) $(TARGET_ARCH)

all: main

domain.o: domain.cpp domain.h

main.o: main.cpp $(HEADERS)

main: main.o domain.o

clean:
	rm *.o



