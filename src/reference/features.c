/*
 * features.c - deterministic sequential feature kernels.
 *
 * Every loop visits nodes and neighbours in ascending ID order, so the
 * floating-point results are the same on every run and every machine that
 * uses IEEE-754 doubles without -ffast-math.
 */
#include "features.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void features_alloc(features_t *f, int32_t n)
{
    memset(f, 0, sizeof(*f));
    size_t N = (size_t)n;
    f->in_degree         = xcalloc(N, sizeof(int32_t));
    f->out_degree        = xcalloc(N, sizeof(int32_t));
    f->total_degree      = xcalloc(N, sizeof(int32_t));
    f->undirected_degree = xcalloc(N, sizeof(int32_t));
    f->nbr_deg_mean      = xcalloc(N, sizeof(double));
    f->nbr_deg_max       = xcalloc(N, sizeof(int32_t));
    f->two_hop_reach     = xcalloc(N, sizeof(int32_t));
    f->triangles         = xcalloc(N, sizeof(int64_t));
    f->clustering        = xcalloc(N, sizeof(double));
    f->component_size    = xcalloc(N, sizeof(int32_t));
    f->pagerank          = xcalloc(N, sizeof(double));
    f->pr_damping = PR_DAMPING_DEFAULT;
    f->pr_iters   = PR_ITERS_DEFAULT;
}

void features_free(features_t *f)
{
    free(f->in_degree); free(f->out_degree); free(f->total_degree);
    free(f->undirected_degree); free(f->nbr_deg_mean); free(f->nbr_deg_max);
    free(f->two_hop_reach); free(f->triangles); free(f->clustering);
    free(f->component_size); free(f->pagerank);
    memset(f, 0, sizeof(*f));
}

/* 1-4: in, out, total, undirected degree. O(N). */
void feat_degrees(const graph_t *g, features_t *f)
{
    for (int32_t v = 0; v < g->n; v++) {
        int32_t in  = (int32_t)(g->in_ptr[v + 1]  - g->in_ptr[v]);
        int32_t out = (int32_t)(g->out_ptr[v + 1] - g->out_ptr[v]);
        f->in_degree[v] = in;
        f->out_degree[v] = out;
        f->total_degree[v] = in + out;
        f->undirected_degree[v] = (int32_t)(g->und_ptr[v + 1] - g->und_ptr[v]);
    }
}

/* 5-6: mean and max undirected degree of neighbours. O(N + M_u).
 * The sum is an exact integer, divided once, so the result is bit-exact. */
void feat_neighbor_degree(const graph_t *g, features_t *f)
{
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

/* 7: distinct nodes within undirected distance 2, excluding v.
 * Stamp array: mark[x] == v means x already counted for v. O(S). */
void feat_two_hop(const graph_t *g, features_t *f)
{
    int32_t *mark = xmalloc((size_t)g->n * sizeof(int32_t));
    for (int32_t i = 0; i < g->n; i++) mark[i] = -1;

    for (int32_t v = 0; v < g->n; v++) {
        int32_t count = 0;
        mark[v] = v;
        for (int64_t k = g->und_ptr[v]; k < g->und_ptr[v + 1]; k++) {
            int32_t u = g->und_idx[k];
            if (mark[u] != v) { mark[u] = v; count++; }
            for (int64_t j = g->und_ptr[u]; j < g->und_ptr[u + 1]; j++) {
                int32_t w = g->und_idx[j];
                if (mark[w] != v) { mark[w] = v; count++; }
            }
        }
        f->two_hop_reach[v] = count;
    }
    free(mark);
}

/* Size of the intersection of two sorted rows. */
static int64_t intersect_count(const int32_t *a, int64_t na, const int32_t *b, int64_t nb)
{
    int64_t i = 0, j = 0, c = 0;
    while (i < na && j < nb) {
        if (a[i] < b[j]) i++;
        else if (b[j] < a[i]) j++;
        else { c++; i++; j++; }
    }
    return c;
}

/* 8-9: triangles through v and local clustering coefficient.
 * For each neighbour u, |N(v) ∩ N(u)| counts each triangle {v,u,w} once
 * from u and once from w, so the sum is divided by 2. O(S). */
void feat_triangles(const graph_t *g, features_t *f)
{
    for (int32_t v = 0; v < g->n; v++) {
        const int32_t *nv = g->und_idx + g->und_ptr[v];
        int64_t dv = g->und_ptr[v + 1] - g->und_ptr[v];
        int64_t twice = 0;
        for (int64_t k = 0; k < dv; k++) {
            int32_t u = nv[k];
            twice += intersect_count(nv, dv, g->und_idx + g->und_ptr[u],
                                     g->und_ptr[u + 1] - g->und_ptr[u]);
        }
        int64_t t = twice / 2;
        f->triangles[v] = t;
        f->clustering[v] = dv >= 2 ? (2.0 * (double)t) / ((double)dv * (double)(dv - 1)) : 0.0;
    }
}

/* 10: connected-component size in U_t (BFS). O(N + M_u). */
void feat_components(const graph_t *g, features_t *f)
{
    int32_t n = g->n;
    int32_t *comp  = xmalloc((size_t)n * sizeof(int32_t));
    int32_t *queue = xmalloc((size_t)n * sizeof(int32_t));
    int32_t *size  = xcalloc((size_t)n, sizeof(int32_t));
    for (int32_t i = 0; i < n; i++) comp[i] = -1;

    for (int32_t s = 0; s < n; s++) {
        if (comp[s] != -1) continue;
        /* component label = smallest node ID in it (s is visited first) */
        int32_t head = 0, tail = 0;
        queue[tail++] = s;
        comp[s] = s;
        while (head < tail) {
            int32_t v = queue[head++];
            for (int64_t k = g->und_ptr[v]; k < g->und_ptr[v + 1]; k++) {
                int32_t u = g->und_idx[k];
                if (comp[u] == -1) { comp[u] = s; queue[tail++] = u; }
            }
        }
        size[s] = tail;
    }
    for (int32_t v = 0; v < n; v++) f->component_size[v] = size[comp[v]];
    free(comp); free(queue); free(size);
}

/* 11: per-step PageRank, pull form, fixed iteration count (spec 3.1).
 * O(K (N + M)). */
void feat_pagerank(const graph_t *g, features_t *f)
{
    int32_t n = g->n;
    int32_t S = g->max_step + 1;
    double a = f->pr_damping;
    int K = f->pr_iters;

    int64_t *n_t  = xcalloc((size_t)S, sizeof(int64_t));
    double  *dang = xcalloc((size_t)S, sizeof(double));
    double  *x    = xmalloc((size_t)n * sizeof(double));
    double  *y    = xmalloc((size_t)n * sizeof(double));
    double  *contrib = xmalloc((size_t)n * sizeof(double));

    for (int32_t v = 0; v < n; v++) n_t[g->step[v]]++;
    for (int32_t v = 0; v < n; v++) x[v] = 1.0 / (double)n_t[g->step[v]];

    double l1 = 0.0;
    for (int it = 0; it < K; it++) {
        memset(dang, 0, (size_t)S * sizeof(double));
        for (int32_t u = 0; u < n; u++) {
            int64_t od = g->out_ptr[u + 1] - g->out_ptr[u];
            if (od == 0) { dang[g->step[u]] += x[u]; contrib[u] = 0.0; }
            else contrib[u] = x[u] / (double)od;
        }
        for (int32_t v = 0; v < n; v++) {
            double sum = 0.0;
            for (int64_t k = g->in_ptr[v]; k < g->in_ptr[v + 1]; k++)
                sum += contrib[g->in_idx[k]];
            double nt = (double)n_t[g->step[v]];
            y[v] = (1.0 - a) / nt + a * (sum + dang[g->step[v]] / nt);
        }
        l1 = 0.0;
        for (int32_t v = 0; v < n; v++) l1 += fabs(y[v] - x[v]);
        double *tmp = x; x = y; y = tmp;
    }
    f->pr_final_l1_delta = K > 0 ? l1 : 0.0;

    for (int32_t v = 0; v < n; v++)
        f->pagerank[v] = x[v] * (double)n_t[g->step[v]];

    free(n_t); free(dang); free(x); free(y); free(contrib);
}
