/*
 * DAA Lab Assignment
 * Performance Comparison of Traditional and Strassen Matrix Multiplication
 * Covers: Q1 (Implementation), Q2 (Verification), Q3 (Performance),
 *         Q5 (Hybrid/Cutoff Strassen)
 * Compile: gcc -O0 -o matrix matrix_multiplication.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ────────────────────────────────────────────────────────────────
   Memory helpers
   ──────────────────────────────────────────────────────────────── */

double **alloc_matrix(int n) {
    double **m = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        m[i] = (double *)calloc(n, sizeof(double));
    }
    return m;
}

void free_matrix(double **m, int n) {
    for (int i = 0; i < n; i++) free(m[i]);
    free(m);
}

void fill_random(double **m, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            m[i][j] = (double)(rand() % 10 + 1);   /* values 1‑10 */
}

void copy_matrix(double **src, double **dst, int n) {
    for (int i = 0; i < n; i++)
        memcpy(dst[i], src[i], n * sizeof(double));
}

void print_matrix(double **m, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            printf("%8.1f ", m[i][j]);
        printf("\n");
    }
}

/* ────────────────────────────────────────────────────────────────
   Q1-a  Traditional Matrix Multiplication  O(n^3)
   ──────────────────────────────────────────────────────────────── */

void traditional_multiply(double **A, double **B, double **C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < n; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
}

/* ────────────────────────────────────────────────────────────────
   Strassen helpers: add / subtract sub-matrices
   ──────────────────────────────────────────────────────────────── */

void add_matrix(double **A, double **B, double **C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] + B[i][j];
}

void sub_matrix(double **A, double **B, double **C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] - B[i][j];
}

/* Copy a sub-block of size half x half from src[rOff][cOff] into dst */
void get_submatrix(double **src, double **dst, int rOff, int cOff, int half) {
    for (int i = 0; i < half; i++)
        for (int j = 0; j < half; j++)
            dst[i][j] = src[i + rOff][j + cOff];
}

/* Write dst back into a sub-block of res[rOff][cOff] */
void set_submatrix(double **res, double **src, int rOff, int cOff, int half) {
    for (int i = 0; i < half; i++)
        for (int j = 0; j < half; j++)
            res[i + rOff][j + cOff] = src[i][j];
}

/* ────────────────────────────────────────────────────────────────
   Q1-b  Strassen Matrix Multiplication  O(n^2.807)
   ──────────────────────────────────────────────────────────────── */

void strassen(double **A, double **B, double **C, int n) {
    if (n == 1) {
        C[0][0] = A[0][0] * B[0][0];
        return;
    }

    int half = n / 2;

    /* Allocate sub-matrices */
    double **A11 = alloc_matrix(half), **A12 = alloc_matrix(half);
    double **A21 = alloc_matrix(half), **A22 = alloc_matrix(half);
    double **B11 = alloc_matrix(half), **B12 = alloc_matrix(half);
    double **B21 = alloc_matrix(half), **B22 = alloc_matrix(half);

    double **M1 = alloc_matrix(half), **M2 = alloc_matrix(half);
    double **M3 = alloc_matrix(half), **M4 = alloc_matrix(half);
    double **M5 = alloc_matrix(half), **M6 = alloc_matrix(half);
    double **M7 = alloc_matrix(half);

    double **T1 = alloc_matrix(half), **T2 = alloc_matrix(half);

    /* Split A and B into quadrants */
    get_submatrix(A, A11, 0,    0,    half);
    get_submatrix(A, A12, 0,    half, half);
    get_submatrix(A, A21, half, 0,    half);
    get_submatrix(A, A22, half, half, half);

    get_submatrix(B, B11, 0,    0,    half);
    get_submatrix(B, B12, 0,    half, half);
    get_submatrix(B, B21, half, 0,    half);
    get_submatrix(B, B22, half, half, half);

    /* M1 = (A11 + A22) * (B11 + B22) */
    add_matrix(A11, A22, T1, half);
    add_matrix(B11, B22, T2, half);
    strassen(T1, T2, M1, half);

    /* M2 = (A21 + A22) * B11 */
    add_matrix(A21, A22, T1, half);
    strassen(T1, B11, M2, half);

    /* M3 = A11 * (B12 - B22) */
    sub_matrix(B12, B22, T1, half);
    strassen(A11, T1, M3, half);

    /* M4 = A22 * (B21 - B11) */
    sub_matrix(B21, B11, T1, half);
    strassen(A22, T1, M4, half);

    /* M5 = (A11 + A12) * B22 */
    add_matrix(A11, A12, T1, half);
    strassen(T1, B22, M5, half);

    /* M6 = (A21 - A11) * (B11 + B12) */
    sub_matrix(A21, A11, T1, half);
    add_matrix(B11, B12, T2, half);
    strassen(T1, T2, M6, half);

    /* M7 = (A12 - A22) * (B21 + B22) */
    sub_matrix(A12, A22, T1, half);
    add_matrix(B21, B22, T2, half);
    strassen(T1, T2, M7, half);

    /* C11 = M1 + M4 - M5 + M7 */
    add_matrix(M1, M4, T1, half);
    sub_matrix(T1, M5, T2, half);
    add_matrix(T2, M7, T1, half);
    set_submatrix(C, T1, 0, 0, half);

    /* C12 = M3 + M5 */
    add_matrix(M3, M5, T1, half);
    set_submatrix(C, T1, 0, half, half);

    /* C21 = M2 + M4 */
    add_matrix(M2, M4, T1, half);
    set_submatrix(C, T1, half, 0, half);

    /* C22 = M1 - M2 + M3 + M6 */
    sub_matrix(M1, M2, T1, half);
    add_matrix(T1, M3, T2, half);
    add_matrix(T2, M6, T1, half);
    set_submatrix(C, T1, half, half, half);

    /* Free everything */
    double **ptrs[] = {A11,A12,A21,A22,B11,B12,B21,B22,M1,M2,M3,M4,M5,M6,M7,T1,T2};
    for (int i = 0; i < 17; i++) free_matrix(ptrs[i], half);
}

/* ────────────────────────────────────────────────────────────────
   Q5  Hybrid Strassen with cutoff
   ──────────────────────────────────────────────────────────────── */

void hybrid_strassen(double **A, double **B, double **C, int n, int cutoff) {
    if (n <= cutoff) {
        traditional_multiply(A, B, C, n);
        return;
    }

    int half = n / 2;

    double **A11 = alloc_matrix(half), **A12 = alloc_matrix(half);
    double **A21 = alloc_matrix(half), **A22 = alloc_matrix(half);
    double **B11 = alloc_matrix(half), **B12 = alloc_matrix(half);
    double **B21 = alloc_matrix(half), **B22 = alloc_matrix(half);
    double **M1  = alloc_matrix(half), **M2  = alloc_matrix(half);
    double **M3  = alloc_matrix(half), **M4  = alloc_matrix(half);
    double **M5  = alloc_matrix(half), **M6  = alloc_matrix(half);
    double **M7  = alloc_matrix(half);
    double **T1  = alloc_matrix(half), **T2  = alloc_matrix(half);

    get_submatrix(A, A11, 0,    0,    half);
    get_submatrix(A, A12, 0,    half, half);
    get_submatrix(A, A21, half, 0,    half);
    get_submatrix(A, A22, half, half, half);
    get_submatrix(B, B11, 0,    0,    half);
    get_submatrix(B, B12, 0,    half, half);
    get_submatrix(B, B21, half, 0,    half);
    get_submatrix(B, B22, half, half, half);

    add_matrix(A11, A22, T1, half); add_matrix(B11, B22, T2, half);
    hybrid_strassen(T1, T2, M1, half, cutoff);

    add_matrix(A21, A22, T1, half);
    hybrid_strassen(T1, B11, M2, half, cutoff);

    sub_matrix(B12, B22, T1, half);
    hybrid_strassen(A11, T1, M3, half, cutoff);

    sub_matrix(B21, B11, T1, half);
    hybrid_strassen(A22, T1, M4, half, cutoff);

    add_matrix(A11, A12, T1, half);
    hybrid_strassen(T1, B22, M5, half, cutoff);

    sub_matrix(A21, A11, T1, half); add_matrix(B11, B12, T2, half);
    hybrid_strassen(T1, T2, M6, half, cutoff);

    sub_matrix(A12, A22, T1, half); add_matrix(B21, B22, T2, half);
    hybrid_strassen(T1, T2, M7, half, cutoff);

    add_matrix(M1, M4, T1, half); sub_matrix(T1, M5, T2, half);
    add_matrix(T2, M7, T1, half); set_submatrix(C, T1, 0, 0, half);

    add_matrix(M3, M5, T1, half); set_submatrix(C, T1, 0, half, half);
    add_matrix(M2, M4, T1, half); set_submatrix(C, T1, half, 0, half);

    sub_matrix(M1, M2, T1, half); add_matrix(T1, M3, T2, half);
    add_matrix(T2, M6, T1, half); set_submatrix(C, T1, half, half, half);

    double **ptrs[] = {A11,A12,A21,A22,B11,B12,B21,B22,M1,M2,M3,M4,M5,M6,M7,T1,T2};
    for (int i = 0; i < 17; i++) free_matrix(ptrs[i], half);
}

/* ────────────────────────────────────────────────────────────────
   Q2  Output Verification
   ──────────────────────────────────────────────────────────────── */

int verify_output(double **C_T, double **C_S, int n) {
    double eps = 1e-6;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (C_T[i][j] - C_S[i][j] > eps || C_S[i][j] - C_T[i][j] > eps)
                return 0;
    return 1;
}

/* ────────────────────────────────────────────────────────────────
   Timing helper
   ──────────────────────────────────────────────────────────────── */

double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ────────────────────────────────────────────────────────────────
   MAIN
   ──────────────────────────────────────────────────────────────── */

int main(void) {
    srand(42);   /* fixed seed — same matrices every run */

    /* ── Q1: Display result for small matrix (4x4) ──────────────── */
    printf("=================================================\n");
    printf("  Q1 – Small Matrix Display (4x4)\n");
    printf("=================================================\n");
    {
        int n = 4;
        double **A   = alloc_matrix(n);
        double **B   = alloc_matrix(n);
        double **C_T = alloc_matrix(n);
        double **C_S = alloc_matrix(n);

        fill_random(A, n);
        fill_random(B, n);

        printf("\nMatrix A:\n"); print_matrix(A, n);
        printf("\nMatrix B:\n"); print_matrix(B, n);

        traditional_multiply(A, B, C_T, n);
        strassen(A, B, C_S, n);

        printf("\nResult C_T (Traditional):\n"); print_matrix(C_T, n);
        printf("\nResult C_S (Strassen):\n");    print_matrix(C_S, n);

        free_matrix(A,n); free_matrix(B,n);
        free_matrix(C_T,n); free_matrix(C_S,n);
    }

    /* ── Q2: Output Verification ─────────────────────────────────── */
    printf("\n=================================================\n");
    printf("  Q2 – Output Verification\n");
    printf("=================================================\n");
    printf("%-12s  %-25s\n", "Matrix Size", "Output Verification");
    printf("%-12s  %-25s\n", "-----------", "-------------------");

    int test_sizes[] = {4, 8, 16, 32};
    for (int t = 0; t < 4; t++) {
        int n = test_sizes[t];
        double **A   = alloc_matrix(n);
        double **B   = alloc_matrix(n);
        double **C_T = alloc_matrix(n);
        double **C_S = alloc_matrix(n);

        srand(42);
        fill_random(A, n);
        fill_random(B, n);

        traditional_multiply(A, B, C_T, n);
        strassen(A, B, C_S, n);

        int ok = verify_output(C_T, C_S, n);
        printf("Test %-2d (%-3d)  %s\n", t+1, n,
               ok ? "Output Verification: SUCCESS" : "Output Verification: FAILED");

        free_matrix(A,n); free_matrix(B,n);
        free_matrix(C_T,n); free_matrix(C_S,n);
    }

    /* ── Q3: Performance Experiment ──────────────────────────────── */
    printf("\n=================================================\n");
    printf("  Q3 – Performance Experiment (5 runs each)\n");
    printf("=================================================\n");

    int perf_sizes[] = {32, 64, 128, 256};
    int   num_sizes  = 4;
    int   num_runs   = 5;

    printf("\n--- Traditional Matrix Multiplication ---\n");
    printf("%-6s  %8s  %8s  %8s  %8s  %8s  %10s\n",
           "Size","Run1","Run2","Run3","Run4","Run5","Average");

    double trad_avg[4], strass_avg[4];

    for (int s = 0; s < num_sizes; s++) {
        int    n    = perf_sizes[s];
        double sum  = 0.0;
        double runs[5];

        double **A   = alloc_matrix(n);
        double **B   = alloc_matrix(n);
        double **C_T = alloc_matrix(n);

        srand(42); fill_random(A, n); fill_random(B, n);

        for (int r = 0; r < num_runs; r++) {
            double t0 = get_time_ms();
            traditional_multiply(A, B, C_T, n);
            runs[r] = get_time_ms() - t0;
            sum += runs[r];
        }
        trad_avg[s] = sum / num_runs;

        printf("%-6d  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f  %10.3f  ms\n",
               n, runs[0], runs[1], runs[2], runs[3], runs[4], trad_avg[s]);

        free_matrix(A,n); free_matrix(B,n); free_matrix(C_T,n);
    }

    printf("\n--- Strassen Matrix Multiplication ---\n");
    printf("%-6s  %8s  %8s  %8s  %8s  %8s  %10s\n",
           "Size","Run1","Run2","Run3","Run4","Run5","Average");

    for (int s = 0; s < num_sizes; s++) {
        int    n   = perf_sizes[s];
        double sum = 0.0;
        double runs[5];

        double **A   = alloc_matrix(n);
        double **B   = alloc_matrix(n);
        double **C_S = alloc_matrix(n);

        srand(42); fill_random(A, n); fill_random(B, n);

        for (int r = 0; r < num_runs; r++) {
            double t0 = get_time_ms();
            strassen(A, B, C_S, n);
            runs[r] = get_time_ms() - t0;
            sum += runs[r];
        }
        strass_avg[s] = sum / num_runs;

        printf("%-6d  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f  %10.3f  ms\n",
               n, runs[0], runs[1], runs[2], runs[3], runs[4], strass_avg[s]);

        free_matrix(A,n); free_matrix(B,n); free_matrix(C_S,n);
    }

    printf("\n--- Summary: Faster Algorithm per Size ---\n");
    printf("%-6s  %12s  %12s  %-15s\n", "Size","Traditional","Strassen","Faster");
    for (int s = 0; s < num_sizes; s++) {
        printf("%-6d  %11.3f  %11.3f  %s\n",
               perf_sizes[s], trad_avg[s], strass_avg[s],
               trad_avg[s] <= strass_avg[s] ? "Traditional" : "Strassen");
    }

    /* ── Q5: Hybrid Strassen ──────────────────────────────────────── */
    printf("\n=================================================\n");
    printf("  Q5 – Hybrid Strassen with Cutoff\n");
    printf("=================================================\n");

    int cutoffs[] = {16, 32};

    for (int ci = 0; ci < 2; ci++) {
        int cutoff = cutoffs[ci];
        printf("\nCutoff = %d\n", cutoff);
        printf("%-6s  %14s  %14s  %14s\n",
               "Size", "Orig Strassen", "Hybrid Strassen", "Traditional");

        for (int s = 0; s < num_sizes; s++) {
            int n = perf_sizes[s];

            double **A  = alloc_matrix(n);
            double **B  = alloc_matrix(n);
            double **CS = alloc_matrix(n);
            double **CH = alloc_matrix(n);
            double **CT = alloc_matrix(n);

            srand(42); fill_random(A,n); fill_random(B,n);

            /* Original Strassen */
            double t0 = get_time_ms();
            strassen(A, B, CS, n);
            double ts = get_time_ms() - t0;

            /* Hybrid Strassen */
            t0 = get_time_ms();
            hybrid_strassen(A, B, CH, n, cutoff);
            double th = get_time_ms() - t0;

            /* Traditional */
            t0 = get_time_ms();
            traditional_multiply(A, B, CT, n);
            double tt = get_time_ms() - t0;

            /* Verify hybrid is correct */
            int ok = verify_output(CT, CH, n);

            printf("%-6d  %14.3f  %14.3f  %14.3f  [Hybrid %s]\n",
                   n, ts, th, tt, ok ? "CORRECT" : "WRONG");

            free_matrix(A,n); free_matrix(B,n);
            free_matrix(CS,n); free_matrix(CH,n); free_matrix(CT,n);
        }
    }

    printf("\n=================================================\n");
    printf("  All experiments complete.\n");
    printf("=================================================\n");

    return 0;
}
