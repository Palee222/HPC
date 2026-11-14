/*
 * graphfeat - Role 2 sequential reference for structural graph features.
 *
 * Usage:
 *   graphfeat --mapping data/transaction_mapping.csv \
 *             --edges   data/node_edges.csv \
 *             --out     results/graph_features.csv \
 *            [--meta    results/graph_features.meta.json] \
 *            [--pr-iters 200] [--damping 0.85]
 *
 * Output contract: docs/feature_spec.md section 5.
 */
#ifdef __APPLE__
#define _DARWIN_C_SOURCE /* exposes ru_maxrss alongside _POSIX_C_SOURCE */
#endif
#include "features.h"
#include "graph.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

/* Peak resident memory in MiB (ru_maxrss is bytes on macOS, KiB on Linux). */
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

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void usage(void)
{
    fprintf(stderr,
        "usage: graphfeat --mapping FILE --edges FILE --out FILE\n"
        "                 [--meta FILE] [--pr-iters K] [--damping A]\n");
    exit(2);
}

static const char *HEADER =
    "node_id,time_step,in_degree,out_degree,total_degree,undirected_degree,"
    "nbr_deg_mean,nbr_deg_max,two_hop_reach,triangles,clustering,"
    "component_size,pagerank\n";

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

enum { T_LOAD, T_DEG, T_NBR, T_TWOHOP, T_TRI, T_COMP, T_PR, T_WRITE, T_N };
static const char *TNAMES[T_N] = {
    "load_and_build_csr", "degrees", "neighbor_degree", "two_hop",
    "triangles", "components", "pagerank", "write_csv"
};

static void write_meta(const char *path, const graph_t *g, const features_t *f,
                       const double *t, double total)
{
    FILE *o = fopen(path, "wb");
    if (!o) die("cannot open meta '%s': %s", path, strerror(errno));
    fprintf(o, "{\n");
    fprintf(o, "  \"feature_contract_version\": \"%s\",\n", FEATURE_CONTRACT_VERSION);
    fprintf(o, "  \"implementation\": \"sequential-reference\",\n");
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
    const char *mapping = NULL, *edges = NULL, *out = NULL, *meta = NULL;
    double damping = PR_DAMPING_DEFAULT;
    int iters = PR_ITERS_DEFAULT;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (i + 1 >= argc) usage();
        if      (!strcmp(a, "--mapping"))  mapping = argv[++i];
        else if (!strcmp(a, "--edges"))    edges = argv[++i];
        else if (!strcmp(a, "--out"))      out = argv[++i];
        else if (!strcmp(a, "--meta"))     meta = argv[++i];
        else if (!strcmp(a, "--pr-iters")) iters = atoi(argv[++i]);
        else if (!strcmp(a, "--damping"))  damping = atof(argv[++i]);
        else usage();
    }
    if (!mapping || !edges || !out) usage();
    if (iters < 0) die("--pr-iters must be >= 0");
    if (!(damping >= 0.0 && damping < 1.0)) die("--damping must be in [0, 1)");

    double t[T_N] = {0}, t0 = now_sec(), s;
    graph_t g;
    features_t f;

    s = now_sec(); graph_load(&g, mapping, edges);   t[T_LOAD]   = now_sec() - s;
    features_alloc(&f, g.n);
    f.pr_damping = damping;
    f.pr_iters = iters;
    s = now_sec(); feat_degrees(&g, &f);             t[T_DEG]    = now_sec() - s;
    s = now_sec(); feat_neighbor_degree(&g, &f);     t[T_NBR]    = now_sec() - s;
    s = now_sec(); feat_two_hop(&g, &f);             t[T_TWOHOP] = now_sec() - s;
    s = now_sec(); feat_triangles(&g, &f);           t[T_TRI]    = now_sec() - s;
    s = now_sec(); feat_components(&g, &f);          t[T_COMP]   = now_sec() - s;
    s = now_sec(); feat_pagerank(&g, &f);            t[T_PR]     = now_sec() - s;
    s = now_sec(); write_csv(out, &g, &f);           t[T_WRITE]  = now_sec() - s;
    double total = now_sec() - t0;

    if (meta) write_meta(meta, &g, &f, t, total);

    fprintf(stderr,
        "graphfeat: %d nodes, %d steps, %lld edges read -> %lld directed / %lld undirected "
        "(dropped: %lld cross-step, %lld self-loops, %lld duplicates); "
        "pagerank L1 delta %.2e; total %.3f s\n",
        g.n, g.n_steps, (long long)g.edges_read, (long long)g.m, (long long)g.m_und,
        (long long)g.cross_step_dropped, (long long)g.self_loops_dropped,
        (long long)g.duplicates_dropped, f.pr_final_l1_delta, total);

    features_free(&f);
    graph_free(&g);
    return 0;
}
