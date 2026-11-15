#ifdef __APPLE__
#define _DARWIN_C_SOURCE
#endif

#include "../src/reference/features.h"
#include "../src/reference/graph.h"

#include <mpi.h>
#include <omp.h>

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>

#ifndef _OPENMP
#error "Compile this file with OpenMP enabled (for example, -fopenmp)."
#endif

enum {
    T_LOAD, T_DEG, T_NBR, T_TWOHOP, T_TRI, T_COMP, T_PR,
    T_REDUCE, T_WRITE, T_N
};
static const char *TNAMES[T_N] = {
    "load_and_broadcast_graph", "degrees", "neighbor_degree", "two_hop",
    "triangles", "components", "pagerank", "mpi_reduce_results", "write_csv"
};
static const char *HEADER =
    "node_id,time_step,in_degree,out_degree,total_degree,undirected_degree,"
    "nbr_deg_mean,nbr_deg_max,two_hop_reach,triangles,clustering,"
    "component_size,pagerank\n";

typedef struct {
    int32_t step;
    int64_t count;
    double estimated_work;
} step_work_t;

static void mpi_check(int result, const char *call, int rank)
{
    if (result == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(result, message, &length);
    fprintf(stderr, "rank %d: %s failed: %.*s\n", rank, call, length, message);
    MPI_Abort(MPI_COMM_WORLD, result);
}

#define MPI_CALL(call) mpi_check((call), #call, rank)

static void usage(int rank)
{
    if (rank == 0) {
        fprintf(stderr,
            "usage: mpi_graphfeat --mapping FILE --edges FILE --out FILE\n"
            "                    [--meta FILE] [--pr-iters K] [--damping A]\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 2);
}

static double peak_rss_mib(void)
{
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return -1.0;
#ifdef __APPLE__
    return (double)ru.ru_maxrss / (1024.0 * 1024.0);
#else
    return (double)ru.ru_maxrss / 1024.0;
#endif
}

static void bcast_array(void *data, int64_t count, MPI_Datatype type, int rank)
{
    int element_size = 0;
    MPI_CALL(MPI_Type_size(type, &element_size));
    char *bytes = data;
    int64_t offset = 0;
    const int64_t max_chunk = 1 << 26;
    while (offset < count) {
        int chunk = (int)((count - offset) < max_chunk
                          ? count - offset : max_chunk);
        MPI_CALL(MPI_Bcast(bytes + offset * element_size, chunk, type, 0,
                           MPI_COMM_WORLD));
        offset += chunk;
    }
}

static void graph_broadcast(graph_t *g, const char *mapping, const char *edges,
                            int rank)
{
    if (rank == 0) graph_load(g, mapping, edges);

    MPI_CALL(MPI_Bcast(&g->n, 1, MPI_INT32_T, 0, MPI_COMM_WORLD));
    MPI_CALL(MPI_Bcast(&g->max_step, 1, MPI_INT32_T, 0, MPI_COMM_WORLD));
    MPI_CALL(MPI_Bcast(&g->n_steps, 1, MPI_INT32_T, 0, MPI_COMM_WORLD));
    MPI_CALL(MPI_Bcast(&g->m, 1, MPI_INT64_T, 0, MPI_COMM_WORLD));
    MPI_CALL(MPI_Bcast(&g->m_und, 1, MPI_INT64_T, 0, MPI_COMM_WORLD));
    MPI_CALL(MPI_Bcast(&g->mapping_rows, 1, MPI_INT64_T, 0, MPI_COMM_WORLD));
    MPI_CALL(MPI_Bcast(&g->edges_read, 1, MPI_INT64_T, 0, MPI_COMM_WORLD));
    MPI_CALL(MPI_Bcast(&g->cross_step_dropped, 1, MPI_INT64_T, 0, MPI_COMM_WORLD));
    MPI_CALL(MPI_Bcast(&g->self_loops_dropped, 1, MPI_INT64_T, 0, MPI_COMM_WORLD));
    MPI_CALL(MPI_Bcast(&g->duplicates_dropped, 1, MPI_INT64_T, 0, MPI_COMM_WORLD));

    if (rank != 0) {
        g->step = xmalloc((size_t)g->n * sizeof(*g->step));
        g->out_ptr = xmalloc(((size_t)g->n + 1) * sizeof(*g->out_ptr));
        g->out_idx = xmalloc((size_t)g->m * sizeof(*g->out_idx));
        g->in_ptr = xmalloc(((size_t)g->n + 1) * sizeof(*g->in_ptr));
        g->in_idx = xmalloc((size_t)g->m * sizeof(*g->in_idx));
        g->und_ptr = xmalloc(((size_t)g->n + 1) * sizeof(*g->und_ptr));
        g->und_idx = xmalloc((size_t)(2 * g->m_und) * sizeof(*g->und_idx));
    }

    bcast_array(g->step, g->n, MPI_INT32_T, rank);
    bcast_array(g->out_ptr, (int64_t)g->n + 1, MPI_INT64_T, rank);
    bcast_array(g->out_idx, g->m, MPI_INT32_T, rank);
    bcast_array(g->in_ptr, (int64_t)g->n + 1, MPI_INT64_T, rank);
    bcast_array(g->in_idx, g->m, MPI_INT32_T, rank);
    bcast_array(g->und_ptr, (int64_t)g->n + 1, MPI_INT64_T, rank);
    bcast_array(g->und_idx, 2 * g->m_und, MPI_INT32_T, rank);
}

static int compare_steps(const void *left, const void *right)
{
    const step_work_t *a = left;
    const step_work_t *b = right;
    if (a->estimated_work != b->estimated_work)
        return a->estimated_work > b->estimated_work ? -1 : 1;
    if (a->count != b->count) return a->count > b->count ? -1 : 1;
    return (a->step > b->step) - (a->step < b->step);
}

static int32_t *assign_steps(const graph_t *g, int ranks,
                             int iterations,
                             int64_t **counts_out, int64_t **ptr_out,
                             int32_t **nodes_out)
{
    int32_t n_steps = g->max_step + 1;
    int64_t *counts = xcalloc((size_t)n_steps, sizeof(*counts));
    double *work_estimates = xcalloc((size_t)n_steps, sizeof(*work_estimates));
    int64_t *ptr = xcalloc((size_t)n_steps + 1, sizeof(*ptr));
    int32_t *nodes = xmalloc((size_t)g->n * sizeof(*nodes));
    int32_t *owner = xmalloc((size_t)n_steps * sizeof(*owner));
    double *loads = xcalloc((size_t)ranks, sizeof(*loads));
    step_work_t *work = xmalloc((size_t)g->n_steps * sizeof(*work));
    int64_t *cursor = xmalloc((size_t)n_steps * sizeof(*cursor));

    for (int32_t v = 0; v < g->n; v++) {
        int32_t step = g->step[v];
        int64_t degree = g->und_ptr[v + 1] - g->und_ptr[v];
        int64_t out_degree = g->out_ptr[v + 1] - g->out_ptr[v];
        counts[step]++;
        work_estimates[step] += 2.0 * (double)degree * (double)degree
                              + (double)iterations * (double)(out_degree + 1)
                              + (double)degree + 1.0;
    }
    for (int32_t s = 0; s < n_steps; s++) {
        ptr[s + 1] = ptr[s] + counts[s];
        owner[s] = -1;
    }
    memcpy(cursor, ptr, (size_t)n_steps * sizeof(*cursor));
    for (int32_t v = 0; v < g->n; v++)
        nodes[cursor[g->step[v]]++] = v;

    int32_t used = 0;
    for (int32_t s = 0; s < n_steps; s++) {
        if (counts[s] > 0) {
            work[used].step = s;
            work[used].count = counts[s];
            work[used].estimated_work = work_estimates[s];
            used++;
        }
    }
    qsort(work, (size_t)used, sizeof(*work), compare_steps);
    for (int32_t i = 0; i < used; i++) {
        int32_t target = 0;
        for (int32_t r = 1; r < ranks; r++)
            if (loads[r] < loads[target]) target = r;
        owner[work[i].step] = target;
        loads[target] += work[i].estimated_work;
    }

    free(loads);
    free(work);
    free(cursor);
    free(work_estimates);
    *counts_out = counts;
    *ptr_out = ptr;
    *nodes_out = nodes;
    return owner;
}

static void feat_degrees_hybrid(const graph_t *g, features_t *f)
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

static void feat_neighbor_hybrid(const graph_t *g, features_t *f,
                                 const int32_t *owner, int rank)
{
    #pragma omp parallel for schedule(static)
    for (int32_t v = 0; v < g->n; v++) {
        if (owner[g->step[v]] != rank) continue;
        int64_t sum = 0;
        int32_t maximum = 0;
        for (int64_t k = g->und_ptr[v]; k < g->und_ptr[v + 1]; k++) {
            int32_t degree = f->undirected_degree[g->und_idx[k]];
            sum += degree;
            if (degree > maximum) maximum = degree;
        }
        int32_t degree = f->undirected_degree[v];
        f->nbr_deg_mean[v] = degree ? (double)sum / (double)degree : 0.0;
        f->nbr_deg_max[v] = maximum;
    }
}

static void feat_two_hop_hybrid(const graph_t *g, features_t *f,
                                const int32_t *owner, int rank)
{
    #pragma omp parallel
    {
        int32_t *mark = xmalloc((size_t)g->n * sizeof(*mark));
        for (int32_t v = 0; v < g->n; v++) mark[v] = -1;

        #pragma omp for schedule(dynamic, 8)
        for (int32_t v = 0; v < g->n; v++) {
            if (owner[g->step[v]] != rank) continue;
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

static void feat_triangles_hybrid(const graph_t *g, features_t *f,
                                  const int32_t *owner, int rank)
{
    #pragma omp parallel for schedule(dynamic, 1)
    for (int32_t v = 0; v < g->n; v++) {
        if (owner[g->step[v]] != rank) continue;
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

static void feat_components_owned(const graph_t *g, features_t *f,
                                  const int32_t *owner, int rank)
{
    int32_t *component = xmalloc((size_t)g->n * sizeof(*component));
    int32_t *queue = xmalloc((size_t)g->n * sizeof(*queue));
    int32_t *sizes = xcalloc((size_t)g->n, sizeof(*sizes));
    for (int32_t v = 0; v < g->n; v++) component[v] = -1;

    for (int32_t start = 0; start < g->n; start++) {
        if (owner[g->step[start]] != rank || component[start] != -1) continue;
        int32_t head = 0, tail = 0;
        queue[tail++] = start;
        component[start] = start;
        while (head < tail) {
            int32_t v = queue[head++];
            for (int64_t k = g->und_ptr[v]; k < g->und_ptr[v + 1]; k++) {
                int32_t u = g->und_idx[k];
                if (owner[g->step[u]] == rank && component[u] == -1) {
                    component[u] = start;
                    queue[tail++] = u;
                }
            }
        }
        sizes[start] = tail;
    }

    for (int32_t v = 0; v < g->n; v++)
        if (owner[g->step[v]] == rank)
            f->component_size[v] = sizes[component[v]];

    free(component);
    free(queue);
    free(sizes);
}

static double feat_pagerank_hybrid(const graph_t *g, features_t *f,
                                   const int32_t *owner, const int64_t *step_ptr,
                                   const int32_t *step_nodes, int rank)
{
    int32_t n_steps = g->max_step + 1;
    double *x = xmalloc((size_t)g->n * sizeof(*x));
    double *y = xmalloc((size_t)g->n * sizeof(*y));
    double *contrib = xmalloc((size_t)g->n * sizeof(*contrib));
    double total_l1 = 0.0;

    for (int32_t step = 0; step < n_steps; step++) {
        if (owner[step] != rank) continue;
        int64_t begin = step_ptr[step], end = step_ptr[step + 1];
        double count = (double)(end - begin);
        for (int64_t i = begin; i < end; i++)
            x[step_nodes[i]] = 1.0 / count;

        double l1 = 0.0;
        for (int iteration = 0; iteration < f->pr_iters; iteration++) {
            #pragma omp parallel for schedule(static)
            for (int64_t i = begin; i < end; i++) {
                int32_t u = step_nodes[i];
                int64_t degree = g->out_ptr[u + 1] - g->out_ptr[u];
                contrib[u] = degree == 0 ? 0.0 : x[u] / (double)degree;
            }

            double dangling = 0.0;
            for (int64_t i = begin; i < end; i++) {
                int32_t u = step_nodes[i];
                if (g->out_ptr[u + 1] == g->out_ptr[u]) dangling += x[u];
            }

            #pragma omp parallel for schedule(static)
            for (int64_t i = begin; i < end; i++) {
                int32_t v = step_nodes[i];
                double sum = 0.0;
                for (int64_t k = g->in_ptr[v]; k < g->in_ptr[v + 1]; k++)
                    sum += contrib[g->in_idx[k]];
                y[v] = (1.0 - f->pr_damping) / count +
                       f->pr_damping * (sum + dangling / count);
            }

            l1 = 0.0;
            for (int64_t i = begin; i < end; i++) {
                int32_t v = step_nodes[i];
                l1 += fabs(y[v] - x[v]);
            }
            double *tmp = x;
            x = y;
            y = tmp;
        }
        total_l1 += l1;

        #pragma omp parallel for schedule(static)
        for (int64_t i = begin; i < end; i++) {
            int32_t v = step_nodes[i];
            f->pagerank[v] = x[v] * count;
        }
    }

    free(x);
    free(y);
    free(contrib);
    return total_l1;
}

static void clear_unowned(const graph_t *g, features_t *f,
                          const int32_t *owner, int rank)
{
    for (int32_t v = 0; v < g->n; v++) {
        if (owner[g->step[v]] == rank) continue;
        f->in_degree[v] = 0;
        f->out_degree[v] = 0;
        f->total_degree[v] = 0;
        f->undirected_degree[v] = 0;
        f->nbr_deg_mean[v] = 0.0;
        f->nbr_deg_max[v] = 0;
        f->two_hop_reach[v] = 0;
        f->triangles[v] = 0;
        f->clustering[v] = 0.0;
        f->component_size[v] = 0;
        f->pagerank[v] = 0.0;
    }
}

static void reduce_array(void *values, int count, MPI_Datatype type, int rank)
{
    void *send = rank == 0 ? MPI_IN_PLACE : values;
    MPI_CALL(MPI_Reduce(send, values, count, type, MPI_SUM, 0, MPI_COMM_WORLD));
}

static void reduce_features(features_t *f, int32_t n, int rank)
{
    int count = (int)n;
    reduce_array(f->in_degree, count, MPI_INT32_T, rank);
    reduce_array(f->out_degree, count, MPI_INT32_T, rank);
    reduce_array(f->total_degree, count, MPI_INT32_T, rank);
    reduce_array(f->undirected_degree, count, MPI_INT32_T, rank);
    reduce_array(f->nbr_deg_mean, count, MPI_DOUBLE, rank);
    reduce_array(f->nbr_deg_max, count, MPI_INT32_T, rank);
    reduce_array(f->two_hop_reach, count, MPI_INT32_T, rank);
    reduce_array(f->triangles, count, MPI_INT64_T, rank);
    reduce_array(f->clustering, count, MPI_DOUBLE, rank);
    reduce_array(f->component_size, count, MPI_INT32_T, rank);
    reduce_array(f->pagerank, count, MPI_DOUBLE, rank);
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
                       const double *t, double total, int ranks, int threads)
{
    FILE *o = fopen(path, "wb");
    if (!o) die("cannot open meta '%s': %s", path, strerror(errno));
    fprintf(o, "{\n");
    fprintf(o, "  \"feature_contract_version\": \"%s\",\n", FEATURE_CONTRACT_VERSION);
    fprintf(o, "  \"implementation\": \"mpi+openmp\",\n");
    fprintf(o, "  \"mpi_ranks\": %d,\n", ranks);
    fprintf(o, "  \"omp_threads_per_rank\": %d,\n", threads);
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

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI implementation lacks MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const char *mapping = NULL, *edges = NULL, *out = NULL, *meta = NULL;
    double damping = PR_DAMPING_DEFAULT;
    int iterations = PR_ITERS_DEFAULT;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (i + 1 >= argc) usage(rank);
        if      (!strcmp(arg, "--mapping"))  mapping = argv[++i];
        else if (!strcmp(arg, "--edges"))    edges = argv[++i];
        else if (!strcmp(arg, "--out"))      out = argv[++i];
        else if (!strcmp(arg, "--meta"))     meta = argv[++i];
        else if (!strcmp(arg, "--pr-iters")) iterations = atoi(argv[++i]);
        else if (!strcmp(arg, "--damping"))  damping = atof(argv[++i]);
        else usage(rank);
    }
    if (!mapping || !edges || !out) usage(rank);
    if (iterations < 0 || !(damping >= 0.0 && damping < 1.0)) usage(rank);

    omp_set_dynamic(0);
    int threads = omp_get_max_threads();
    double timings[T_N] = {0};
    double total_start = MPI_Wtime();
    double start = MPI_Wtime();
    graph_t g;
    memset(&g, 0, sizeof(g));
    graph_broadcast(&g, mapping, edges, rank);
    timings[T_LOAD] = MPI_Wtime() - start;

    int64_t *step_counts = NULL, *step_ptr = NULL;
    int32_t *step_nodes = NULL;
    int32_t *owner = assign_steps(&g, ranks, iterations, &step_counts,
                                  &step_ptr, &step_nodes);
    features_t f;
    features_alloc(&f, g.n);
    f.pr_damping = damping;
    f.pr_iters = iterations;

    start = MPI_Wtime();
    feat_degrees_hybrid(&g, &f);
    timings[T_DEG] = MPI_Wtime() - start;

    start = MPI_Wtime();
    feat_neighbor_hybrid(&g, &f, owner, rank);
    timings[T_NBR] = MPI_Wtime() - start;

    start = MPI_Wtime();
    feat_two_hop_hybrid(&g, &f, owner, rank);
    timings[T_TWOHOP] = MPI_Wtime() - start;

    start = MPI_Wtime();
    feat_triangles_hybrid(&g, &f, owner, rank);
    timings[T_TRI] = MPI_Wtime() - start;

    start = MPI_Wtime();
    feat_components_owned(&g, &f, owner, rank);
    timings[T_COMP] = MPI_Wtime() - start;

    start = MPI_Wtime();
    f.pr_final_l1_delta = feat_pagerank_hybrid(
        &g, &f, owner, step_ptr, step_nodes, rank);
    timings[T_PR] = MPI_Wtime() - start;

    start = MPI_Wtime();
    clear_unowned(&g, &f, owner, rank);
    reduce_features(&f, g.n, rank);

    double global_delta = 0.0;
    double local_delta = f.pr_final_l1_delta;
    if (rank == 0) global_delta = local_delta;
    double *delta_send = rank == 0 ? MPI_IN_PLACE : &local_delta;
    MPI_CALL(MPI_Reduce(delta_send, &global_delta, 1, MPI_DOUBLE, MPI_SUM, 0,
                        MPI_COMM_WORLD));
    if (rank == 0) f.pr_final_l1_delta = global_delta;
    timings[T_REDUCE] = MPI_Wtime() - start;

    double global_timings[T_N] = {0};
    MPI_CALL(MPI_Reduce(timings, global_timings, T_N, MPI_DOUBLE, MPI_MAX, 0,
                        MPI_COMM_WORLD));

    if (rank == 0) {
        start = MPI_Wtime();
        write_csv(out, &g, &f);
        global_timings[T_WRITE] = MPI_Wtime() - start;
        double total = MPI_Wtime() - total_start;
        if (meta)
            write_meta(meta, &g, &f, global_timings, total, ranks, threads);
        fprintf(stderr,
            "mpi_graphfeat: %d nodes, %d steps, %d ranks x %d threads, "
            "%lld directed / %lld undirected edges; total %.3f s\n",
            g.n, g.n_steps, ranks, threads,
            (long long)g.m, (long long)g.m_und, total);
    }

    features_free(&f);
    graph_free(&g);
    free(step_counts);
    free(step_ptr);
    free(step_nodes);
    free(owner);
    MPI_Finalize();
    return 0;
}
