libliharm.so: libagg.o
	gcc -fopenmp -shared -o libiharm.so libagg.o
	-rm *.o

libagg.o: libiharm.o libu2p.o
	ld -r libiharm.o -o libagg.o libu2p.o

libu2p.o: u2p.c
	gcc -c -Wall -Werror -fpic -fopenmp u2p.c -o libu2p.o

libiharm.o: libiharm.c
	gcc -c -Wall -Werror -fpic -fopenmp libiharm.c

clean:
	-rm libiharm.so
	-rm *.o
