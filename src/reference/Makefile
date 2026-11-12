# Role 2 sequential reference.
# No -ffast-math and no -march=native: results must be bit-reproducible
# across runs and machines (docs/feature_spec.md section 6).

CC      ?= cc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra -Wpedantic
CPPFLAGS += -D_POSIX_C_SOURCE=200809L
LDLIBS  += -lm

OBJS = main.o graph.o features.o

graphfeat: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDLIBS)

main.o: main.c graph.h features.h
graph.o: graph.c graph.h
features.o: features.c features.h graph.h

clean:
	rm -f graphfeat $(OBJS)

.PHONY: clean
