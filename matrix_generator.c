// matrix_generator.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <n>\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    FILE *f = fopen("Matrix.txt", "w");
    if (!f) { perror("fopen"); return 1; }
    static char buf[1<<24];
    setvbuf(f, buf, _IOFBF, sizeof(buf));
    srand((unsigned)time(NULL));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float v = ((float)rand() / RAND_MAX) * 10.0f;
            fprintf(f, "%.6f", v);
            if (j+1 < n) fputc(' ', f);
        }
        if (i+1 < n) fputs(",\n", f);
        else fputs(".\n", f);
    }
    fclose(f);

    // generate b.txt
    FILE *b = fopen("b.txt", "w");
    if (!b) { perror("fopen b.txt"); return 1; }
    setvbuf(b, buf, _IOFBF, sizeof(buf));
    for (int i = 0; i < n; i++) {
        float v = ((float)rand() / RAND_MAX) * 10.0f;
        fprintf(b, "%.6f\n", v);
    }
    fclose(b);
    return 0;
}

