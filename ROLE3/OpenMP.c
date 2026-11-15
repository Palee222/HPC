/*
 * OpenMP graph-feature implementation.
 *
 * Build from the repository root with GCC (for example, on the cluster):
 *   gcc -O2 -std=c11 -Wall -Wextra -Wpedantic \
 *       -D_POSIX_C_SOURCE=200809L -fopenmp \
 *       ROLE3/OpenMP.c src/reference/graph.c src/reference/features.c \
 *       -lm -o ROLE3/openmp_graphfeat
 *
 * Apple Clang needs the OpenMP runtime; with Homebrew libomp use:
 *   cc -O2 -std=c11 -Wall -Wextra -Wpedantic \
 *       -D_POSIX_C_SOURCE=200809L -Xpreprocessor -fopenmp \
 *       -I"$(brew --prefix libomp)/include" \
 *       ROLE3/OpenMP.c src/reference/graph.c src/reference/features.c \
 *       -L"$(brew --prefix libomp)/lib" -lomp -lm \
 *       -o ROLE3/openmp_graphfeat
 */
#ifdef __APPLE__
#define _DARWIN_C_SOURCE
#endif

#include "../src/reference/features.h"
#include "../src/reference/graph.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

#ifndef _OPENMP
#error "Compile this file with OpenMP enabled (for example, -fopenmp)."
#endif

static const char *HEADER =
    "node_id,time_step,in_degree,out_degree,total_degree,undirected_degree,"
    "nbr_deg_mean,nbr_deg_max,two_hop_reach,triangles,clustering,"
    "component_size,pagerank\n";

enum { T_LOAD, T_DEG, T_NBR, T_TWOHOP, T_TRI, T_COMP, T_PR, T_WRITE, T_N };
static const char *TNAMES[T_N] = {
    "load_and_build_csr", "degrees", "neighbor_degree", "two_hop",
    "triangles", "components", "pagerank", "write_csv"
};

static double now_sec(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        die("clock_gettime failed: %s", strerror(errno));
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double peak_rss_mib(void)
{
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0)
        return -1.0;
#ifdef __APPLE__
    return (double)ru.ru_maxrss / (1024.0 * 1024.0);
#else
    return (double)ru.ru_maxrss / 1024.0;
#endif
}

static void usage(void)
{
    fprintf(stderr,
        "usage: openmp_graphfeat --mapping FILE --edges FILE --out FILE\n"
        "                       [--meta FILE] [--pr-iters K] [--damping A]\n");
    exit(2);
}

static void write_csv(const char *path, const graph_t *g, const features_t *f)
{
    FILE *o = fopen(path, "wb");
    if (!o) die("cannot open output '%s': %s", path, strerror(errno));
    fputs(HEADER, o);
    for (int32_t v = 0; v < g->n; v++) {
        fprintf(o, "%d,%d,%d,%d,%d,%d,%.17g,%d,%d,%lld,%.17g,%d,%.17g\n",
                v, g->step[v],
                f->in_degree[v], f->out_degree[v], f->total_degree[v],
                f->undirected_degree[v], f->nbr_deg_mean[v], f->nbr_deg_max[v],
                f->two_hop_reach[v], (long long)f->triangles[v], f->clustering[v],
                f->component_size[v], f->pagerank[v]);
    }
    if (fclose(o) != 0) die("error writing '%s'", path);
}

static void write_meta(const char *path, const graph_t *g, const features_t *f,
                       const double *t, double total)
{
    FILE *o = fopen(path, "wb");
    if (!o) die("cannot open meta '%s': %s", path, strerror(errno));
    fprintf(o, "{\n");
    fprintf(o, "  \"feature_contract_version\": \"%s\",\n", FEATURE_CONTRACT_VERSION);
    fprintf(o, "  \"implementation\": \"openmp\",\n");
    fprintf(o, "  \"n_nodes\": %d,\n", g->n);
    fprintf(o, "  \"n_time_steps\": %d,\n", g->n_steps);
    fprintf(o, "  \"mapping_rows\": %lld,\n", (long long)g->mapping_rows);
    fprintf(o, "  \"edges_read\": %lld,\n", (long long)g->edges_read);
    fprintf(o, "  \"cross_step_dropped\": %lld,\n", (long long)g->cross_step_dropped);
    fprintf(o, "  \"self_loops_dropped\": %lld,\n", (long long)g->self_loops_dropped);
    fprintf(o, "  \"duplicates_dropped\": %lld,\n", (long long)g->duplicates_dropped);
    fprintf(o, "  \"directed_edges\": %lld,\n", (long long)g->m);
    fprintf(o, "  \"undirected_edges\": %lld,\n", (long long)g->m_und);
    fprintf(o, "  \"pagerank_damping\": %.17g,\n", f->pr_damping);
    fprintf(o, "  \"pagerank_iterations\": %d,\n", f->pr_iters);
    fprintf(o, "  \"pagerank_final_l1_delta\": %.6e,\n", f->pr_final_l1_delta);
    fprintf(o, "  \"peak_rss_mib\": %.1f,\n", peak_rss_mib());
    fprintf(o, "  \"timings_sec\": {\n");
    for (int i = 0; i < T_N; i++)
        fprintf(o, "    \"%s\": %.6f,\n", TNAMES[i], t[i]);
    fprintf(o, "    \"total\": %.6f\n  }\n}\n", total);
    if (fclose(o) != 0) die("error writing '%s'", path);
}

static void feat_degrees_omp(const graph_t *g, features_t *f)
{
    #pragma omp parallel for schedule(static)
    for (int32_t v = 0; v < g->n; v++) {
        int32_t in = (int32_t)(g->in_ptr[v + 1] - g->in_ptr[v]);
        int32_t out = (int32_t)(g->out_ptr[v + 1] - g->out_ptr[v]);
        f->in_degree[v] = in;
        f->out_degree[v] = out;
        f->total_degree[v] = in + out;
        f->undirected_degree[v] =
            (int32_t)(g->und_ptr[v + 1] - g->und_ptr[v]);
    }
}

static void feat_neighbor_degree_omp(const graph_t *g, features_t *f)
{
    #pragma omp parallel for schedule(static)
    for (int32_t v = 0; v < g->n; v++) {
        int64_t sum = 0;
        int32_t mx = 0;
        for (int64_t k = g->und_ptr[v]; k < g->und_ptr[v + 1]; k++) {
            int32_t du = f->undirected_degree[g->und_idx[k]];
            sum += du;
            if (du > mx) mx = du;
        }
        int32_t d = f->undirected_degree[v];
        f->nbr_deg_mean[v] = d ? (double)sum / (double)d : 0.0;
        f->nbr_deg_max[v] = mx;
    }
}

static void feat_two_hop_omp(const graph_t *g, features_t *f)
{
    #pragma omp parallel
    {
        int32_t *mark = xmalloc((size_t)g->n * sizeof(*mark));
        for (int32_t i = 0; i < g->n; i++) mark[i] = -1;

        #pragma omp for schedule(dynamic, 8)
        for (int32_t v = 0; v < g->n; v++) {
            int32_t count = 0;
            mark[v] = v;
            for (int64_t k = g->und_ptr[v]; k < g->und_ptr[v + 1]; k++) {
                int32_t u = g->und_idx[k];
                if (mark[u] != v) {
                    mark[u] = v;
                    count++;
                }
                for (int64_t j = g->und_ptr[u]; j < g->und_ptr[u + 1]; j++) {
                    int32_t w = g->und_idx[j];
                    if (mark[w] != v) {
                        mark[w] = v;
                        count++;
                    }
                }
            }
            f->two_hop_reach[v] = count;
        }
        free(mark);
    }
}

static int64_t intersect_count(const int32_t *a, int64_t na,
                              const int32_t *b, int64_t nb)
{
    int64_t i = 0, j = 0, count = 0;
    while (i < na && j < nb) {
        if (a[i] < b[j]) i++;
        else if (b[j] < a[i]) j++;
        else {
            count++;
            i++;
            j++;
        }
    }
    return count;
}

static void feat_triangles_omp(const graph_t *g, features_t *f)
{
    #pragma omp parallel for schedule(dynamic, 1)
    for (int32_t v = 0; v < g->n; v++) {
        const int32_t *nv = g->und_idx + g->und_ptr[v];
        int64_t dv = g->und_ptr[v + 1] - g->und_ptr[v];
        int64_t twice = 0;
        for (int64_t k = 0; k < dv; k++) {
            int32_t u = nv[k];
            twice += intersect_count(nv, dv, g->und_idx + g->und_ptr[u],
                                    g->und_ptr[u + 1] - g->und_ptr[u]);
        }
        int64_t triangles = twice / 2;
        f->triangles[v] = triangles;
        f->clustering[v] = dv >= 2
            ? (2.0 * (double)triangles) / ((double)dv * (double)(dv - 1))
            : 0.0;
    }
}

static void feat_pagerank_omp(const graph_t *g, features_t *f)
{
    int32_t n = g->n;
    int32_t n_steps = g->max_step + 1;
    double damping = f->pr_damping;
    int iterations = f->pr_iters;

    int64_t *step_counts = xcalloc((size_t)n_steps, sizeof(*step_counts));
    double *dangling = xcalloc((size_t)n_steps, sizeof(*dangling));
    double *x = xmalloc((size_t)n * sizeof(*x));
    double *y = xmalloc((size_t)n * sizeof(*y));
    double *contrib = xmalloc((size_t)n * sizeof(*contrib));

    for (int32_t v = 0; v < n; v++) step_counts[g->step[v]]++;
    for (int32_t v = 0; v < n; v++)
        x[v] = 1.0 / (double)step_counts[g->step[v]];

    double l1 = 0.0;
    for (int it = 0; it < iterations; it++) {
        memset(dangling, 0, (size_t)n_steps * sizeof(*dangling));

        #pragma omp parallel for schedule(static)
        for (int32_t u = 0; u < n; u++) {
            int64_t out_degree = g->out_ptr[u + 1] - g->out_ptr[u];
            contrib[u] = out_degree == 0 ? 0.0 : x[u] / (double)out_degree;
        }

        /* Keep this accumulation ordered as in the reference implementation. */
        for (int32_t u = 0; u < n; u++) {
            if (g->out_ptr[u + 1] == g->out_ptr[u])
                dangling[g->step[u]] += x[u];
        }

        #pragma omp parallel for schedule(static)
        for (int32_t v = 0; v < n; v++) {
            double sum = 0.0;
            for (int64_t k = g->in_ptr[v]; k < g->in_ptr[v + 1]; k++)
                sum += contrib[g->in_idx[k]];
            double nt = (double)step_counts[g->step[v]];
            y[v] = (1.0 - damping) / nt +
                   damping * (sum + dangling[g->step[v]] / nt);
        }

        l1 = 0.0;
        for (int32_t v = 0; v < n; v++) l1 += fabs(y[v] - x[v]);
        double *tmp = x;
        x = y;
        y = tmp;
    }
    f->pr_final_l1_delta = iterations > 0 ? l1 : 0.0;

    #pragma omp parallel for schedule(static)
    for (int32_t v = 0; v < n; v++)
        f->pagerank[v] = x[v] * (double)step_counts[g->step[v]];

    free(step_counts);
    free(dangling);
    free(x);
    free(y);
    free(contrib);
}

int main(int argc, char **argv)
{
    const char *mapping = NULL, *edges = NULL, *out = NULL, *meta = NULL;
    double damping = PR_DAMPING_DEFAULT;
    int iterations = PR_ITERS_DEFAULT;

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (i + 1 >= argc) usage();
        if      (!strcmp(arg, "--mapping"))  mapping = argv[++i];
        else if (!strcmp(arg, "--edges"))    edges = argv[++i];
        else if (!strcmp(arg, "--out"))      out = argv[++i];
        else if (!strcmp(arg, "--meta"))     meta = argv[++i];
        else if (!strcmp(arg, "--pr-iters")) iterations = atoi(argv[++i]);
        else if (!strcmp(arg, "--damping"))  damping = atof(argv[++i]);
        else usage();
    }
    if (!mapping || !edges || !out) usage();
    if (iterations < 0) die("--pr-iters must be >= 0");
    if (!(damping >= 0.0 && damping < 1.0))
        die("--damping must be in [0, 1)");

    double timings[T_N] = {0};
    double total_start = now_sec();
    double start;
    graph_t g;
    features_t f;

    start = now_sec();
    graph_load(&g, mapping, edges);
    timings[T_LOAD] = now_sec() - start;

    features_alloc(&f, g.n);
    f.pr_damping = damping;
    f.pr_iters = iterations;

    start = now_sec();
    feat_degrees_omp(&g, &f);
    timings[T_DEG] = now_sec() - start;

    start = now_sec();
    feat_neighbor_degree_omp(&g, &f);
    timings[T_NBR] = now_sec() - start;

    start = now_sec();
    feat_two_hop_omp(&g, &f);
    timings[T_TWOHOP] = now_sec() - start;

    start = now_sec();
    feat_triangles_omp(&g, &f);
    timings[T_TRI] = now_sec() - start;

    start = now_sec();
    feat_components(&g, &f);
    timings[T_COMP] = now_sec() - start;

    start = now_sec();
    feat_pagerank_omp(&g, &f);
    timings[T_PR] = now_sec() - start;

    start = now_sec();
    write_csv(out, &g, &f);
    timings[T_WRITE] = now_sec() - start;

    double total = now_sec() - total_start;
    if (meta) write_meta(meta, &g, &f, timings, total);

    fprintf(stderr,
        "openmp_graphfeat: %d nodes, %d steps, %lld edges read -> "
        "%lld directed / %lld undirected "
        "(dropped: %lld cross-step, %lld self-loops, %lld duplicates); "
        "pagerank L1 delta %.2e; total %.3f s\n",
        g.n, g.n_steps, (long long)g.edges_read, (long long)g.m,
        (long long)g.m_und, (long long)g.cross_step_dropped,
        (long long)g.self_loops_dropped, (long long)g.duplicates_dropped,
        f.pr_final_l1_delta, total);

    features_free(&f);
    graph_free(&g);
    return 0;
}
