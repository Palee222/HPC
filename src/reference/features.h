/*
 * features.h - sequential reference kernels for the 11 structural features.
 * Definitions: docs/feature_spec.md section 3.
 */
#ifndef FEATURES_H
#define FEATURES_H

#include "graph.h"

#define FEATURE_CONTRACT_VERSION "role2-v1"
#define PR_DAMPING_DEFAULT 0.85
#define PR_ITERS_DEFAULT 200

typedef struct {
    int32_t *in_degree, *out_degree, *total_degree, *undirected_degree;
    double  *nbr_deg_mean;
    int32_t *nbr_deg_max;
    int32_t *two_hop_reach;
    int64_t *triangles;
    double  *clustering;
    int32_t *component_size;
    double  *pagerank;

    double pr_damping;
    int    pr_iters;
    double pr_final_l1_delta;
} features_t;

void features_alloc(features_t *f, int32_t n);
void features_free(features_t *f);

/* Each kernel fills its own column(s). Order matters only where noted. */
void feat_degrees(const graph_t *g, features_t *f);          /* cols 1-4        */
void feat_neighbor_degree(const graph_t *g, features_t *f);  /* cols 5-6, needs 4 */
void feat_two_hop(const graph_t *g, features_t *f);          /* col 7           */
void feat_triangles(const graph_t *g, features_t *f);        /* cols 8-9        */
void feat_components(const graph_t *g, features_t *f);       /* col 10          */
void feat_pagerank(const graph_t *g, features_t *f);         /* col 11          */

#endif
