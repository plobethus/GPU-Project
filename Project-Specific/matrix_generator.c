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
    if (!f) { perror("fopen Matrix.txt"); return 1; }
    static char buf[1<<24];
    setvbuf(f, buf, _IOFBF, sizeof(buf));
    srand((unsigned)time(NULL));

    // Generate n×n integer matrix, diag ∈ [1..9], off‐diag ∈ [0..9]
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int v;
            if (i == j) {
                // ensure non-zero on the diagonal
                v = (rand() % 9) + 1;  // 1..9
            } else {
                v = rand() % 10;       // 0..9
            }
            fprintf(f, "%d", v);
            if (j + 1 < n) fputc(' ', f);
        }
        if (i + 1 < n)
            fputs(",\n", f);
        else
            fputs(".\n", f);
    }
    fclose(f);

    // Also regenerate b.txt (still 0..9)
    FILE *b = fopen("b.txt", "w");
    if (!b) { perror("fopen b.txt"); return 1; }
    setvbuf(b, buf, _IOFBF, sizeof(buf));
    for (int i = 0; i < n; i++) {
        int v = rand() % 10;  // RHS entries still 0..9
        fprintf(b, "%d\n", v);
    }
    fclose(b);

    return 0;
}