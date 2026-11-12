/*
 * graph.h - input loading and CSR adjacency for the Role 2 reference.
 * See docs/feature_spec.md sections 1, 2 and 4.
 */
#ifndef GRAPH_H
#define GRAPH_H

#include <stddef.h>
#include <stdint.h>

#define MAX_TIME_STEP 100000

typedef struct {
    int32_t n;              /* number of nodes (canonical IDs 0..n-1)   */
    int32_t *step;          /* [n] time step of each node                */
    int32_t max_step;       /* largest time step value                   */
    int32_t n_steps;        /* number of distinct time steps             */

    /* directed simple intra-step graph G_t (all steps together) */
    int64_t m;              /* directed edges after cleaning             */
    int64_t *out_ptr;       /* [n+1] */
    int32_t *out_idx;       /* [m]   sorted per row                      */
    int64_t *in_ptr;        /* [n+1] */
    int32_t *in_idx;        /* [m]   sorted per row                      */

    /* undirected simple graph U_t, each edge stored twice */
    int64_t m_und;          /* undirected edges                          */
    int64_t *und_ptr;       /* [n+1] */
    int32_t *und_idx;       /* [2*m_und] sorted per row                  */

    /* input / cleaning statistics */
    int64_t mapping_rows;
    int64_t edges_read;
    int64_t cross_step_dropped;
    int64_t self_loops_dropped;
    int64_t duplicates_dropped;
} graph_t;

/* Load mapping + edges and build all CSR structures. Exits on any error. */
void graph_load(graph_t *g, const char *mapping_path, const char *edges_path);

void graph_free(graph_t *g);

/* Helpers shared with other modules. */
void *xmalloc(size_t bytes);
void *xcalloc(size_t count, size_t size);
void die(const char *fmt, ...);

#endif
