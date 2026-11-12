/*
 * graph.c - load Role 1's canonical graph and build CSR adjacency.
 *
 * Inputs (docs/feature_spec.md section 1):
 *   transaction_mapping.csv : header with node_id, time_step (other columns ignored)
 *   node_edges.csv          : header with source_node, target_node
 *
 * Everything is deterministic: no hashing, no threads, rows sorted ascending.
 */
#include "graph.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------- helpers */

void die(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "graphfeat: error: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(1);
}

void *xmalloc(size_t bytes)
{
    void *p = malloc(bytes ? bytes : 1);
    if (!p) die("out of memory (%zu bytes)", bytes);
    return p;
}

void *xcalloc(size_t count, size_t size)
{
    void *p = calloc(count ? count : 1, size ? size : 1);
    if (!p) die("out of memory (%zu x %zu bytes)", count, size);
    return p;
}

static int cmp_i32(const void *a, const void *b)
{
    int32_t x = *(const int32_t *)a, y = *(const int32_t *)b;
    return (x > y) - (x < y);
}

/* Growable int32 array. */
typedef struct { int32_t *v; int64_t len, cap; } vec_t;

static void vec_push(vec_t *a, int32_t x)
{
    if (a->len == a->cap) {
        a->cap = a->cap ? a->cap * 2 : 1024;
        a->v = realloc(a->v, (size_t)a->cap * sizeof(int32_t));
        if (!a->v) die("out of memory growing array");
    }
    a->v[a->len++] = x;
}

/* ------------------------------------------------------------ CSV reading */

#define MAX_COLS 64

/* Read one line into buf, strip trailing \r\n. Returns 0 at EOF. */
static int read_line(FILE *f, char **buf, size_t *cap)
{
    size_t len = 0;
    int c;
    if (*cap == 0) { *cap = 256; *buf = xmalloc(*cap); }
    while ((c = fgetc(f)) != EOF) {
        if (c == '\n') break;
        if (len + 1 >= *cap) {
            *cap *= 2;
            *buf = realloc(*buf, *cap);
            if (!*buf) die("out of memory reading line");
        }
        (*buf)[len++] = (char)c;
    }
    if (c == EOF && len == 0) return 0;
    if (len > 0 && (*buf)[len - 1] == '\r') len--;
    (*buf)[len] = '\0';
    return 1;
}

/* Split line in place on commas. Returns number of fields. */
static int split_csv(char *line, char **fields)
{
    int n = 0;
    char *p = line;
    fields[n++] = p;
    for (; *p; p++) {
        if (*p == ',') {
            *p = '\0';
            if (n == MAX_COLS) die("too many columns in CSV line");
            fields[n++] = p + 1;
        }
    }
    return n;
}

static int find_col(char **fields, int nf, const char *name, const char *path)
{
    for (int i = 0; i < nf; i++)
        if (strcmp(fields[i], name) == 0) return i;
    die("%s: header has no column '%s'", path, name);
    return -1;
}

static int64_t parse_int(const char *s, const char *path, int64_t line)
{
    char *end;
    errno = 0;
    long long v = strtoll(s, &end, 10);
    if (errno || end == s || *end != '\0')
        die("%s:%lld: '%s' is not an integer", path, (long long)line, s);
    return (int64_t)v;
}

/* Is this line empty (allow a blank last line)? */
static int is_blank(const char *s) { return s[0] == '\0'; }

static void load_mapping(graph_t *g, const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open mapping file '%s': %s", path, strerror(errno));

    char *buf = NULL; size_t cap = 0;
    char *fields[MAX_COLS];
    if (!read_line(f, &buf, &cap)) die("%s: file is empty", path);
    int nf = split_csv(buf, fields);
    int c_node = find_col(fields, nf, "node_id", path);
    int c_step = find_col(fields, nf, "time_step", path);
    int need = (c_node > c_step ? c_node : c_step) + 1;

    vec_t ids = {0}, steps = {0};
    int64_t line = 1;
    while (read_line(f, &buf, &cap)) {
        line++;
        if (is_blank(buf)) continue;
        nf = split_csv(buf, fields);
        if (nf < need) die("%s:%lld: expected at least %d columns", path, (long long)line, need);
        int64_t id = parse_int(fields[c_node], path, line);
        int64_t st = parse_int(fields[c_step], path, line);
        if (id < 0 || id > INT32_MAX - 1) die("%s:%lld: node_id %lld out of range", path, (long long)line, (long long)id);
        if (st < 0 || st > MAX_TIME_STEP) die("%s:%lld: time_step %lld out of range 0..%d", path, (long long)line, (long long)st, MAX_TIME_STEP);
        vec_push(&ids, (int32_t)id);
        vec_push(&steps, (int32_t)st);
    }
    fclose(f);
    free(buf);

    int64_t rows = ids.len;
    if (rows == 0) die("%s: no data rows", path);
    if (rows > INT32_MAX - 1) die("%s: too many nodes", path);

    g->n = (int32_t)rows;
    g->mapping_rows = rows;
    g->step = xmalloc((size_t)g->n * sizeof(int32_t));
    char *seen = xcalloc((size_t)g->n, 1);
    for (int64_t r = 0; r < rows; r++) {
        int32_t id = ids.v[r];
        if (id >= g->n) die("%s: node_id %d >= number of rows %d (IDs must be 0..N-1)", path, id, g->n);
        if (seen[id]) die("%s: node_id %d appears more than once", path, id);
        seen[id] = 1;
        g->step[id] = steps.v[r];
    }
    /* rows == n and no duplicates => every id 0..n-1 seen exactly once */
    free(seen);
    free(ids.v);
    free(steps.v);

    g->max_step = 0;
    for (int32_t v = 0; v < g->n; v++)
        if (g->step[v] > g->max_step) g->max_step = g->step[v];
    char *present = xcalloc((size_t)g->max_step + 1, 1);
    for (int32_t v = 0; v < g->n; v++) present[g->step[v]] = 1;
    g->n_steps = 0;
    for (int32_t s = 0; s <= g->max_step; s++) g->n_steps += present[s];
    free(present);
}

static void load_edges(graph_t *g, const char *path, vec_t *src, vec_t *dst)
{
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open edges file '%s': %s", path, strerror(errno));

    char *buf = NULL; size_t cap = 0;
    char *fields[MAX_COLS];
    if (!read_line(f, &buf, &cap)) die("%s: file is empty", path);
    int nf = split_csv(buf, fields);
    int c_src = find_col(fields, nf, "source_node", path);
    int c_dst = find_col(fields, nf, "target_node", path);
    int need = (c_src > c_dst ? c_src : c_dst) + 1;

    int64_t line = 1;
    while (read_line(f, &buf, &cap)) {
        line++;
        if (is_blank(buf)) continue;
        nf = split_csv(buf, fields);
        if (nf < need) die("%s:%lld: expected at least %d columns", path, (long long)line, need);
        int64_t u = parse_int(fields[c_src], path, line);
        int64_t v = parse_int(fields[c_dst], path, line);
        if (u < 0 || u >= g->n || v < 0 || v >= g->n)
            die("%s:%lld: edge %lld -> %lld uses a node_id outside 0..%d",
                path, (long long)line, (long long)u, (long long)v, g->n - 1);
        g->edges_read++;
        if (g->step[u] != g->step[v]) { g->cross_step_dropped++; continue; }
        if (u == v) { g->self_loops_dropped++; continue; }
        vec_push(src, (int32_t)u);
        vec_push(dst, (int32_t)v);
    }
    fclose(f);
    free(buf);
}

/* ------------------------------------------------------------- CSR build */

static void build_out_csr(graph_t *g, const vec_t *src, const vec_t *dst)
{
    int32_t n = g->n;
    int64_t me = src->len;
    int64_t *ptr = xcalloc((size_t)n + 1, sizeof(int64_t));
    int32_t *idx = xmalloc((size_t)me * sizeof(int32_t));

    /* counting sort by source */
    for (int64_t e = 0; e < me; e++) ptr[src->v[e] + 1]++;
    for (int32_t v = 0; v < n; v++) ptr[v + 1] += ptr[v];
    int64_t *fill = xmalloc((size_t)n * sizeof(int64_t));
    memcpy(fill, ptr, (size_t)n * sizeof(int64_t));
    for (int64_t e = 0; e < me; e++) idx[fill[src->v[e]]++] = dst->v[e];
    free(fill);

    /* sort each row, drop duplicates, compact in place */
    int64_t w = 0;
    int64_t row_start = 0;
    for (int32_t v = 0; v < n; v++) {
        int64_t a = row_start, b = ptr[v + 1];
        row_start = b;
        qsort(idx + a, (size_t)(b - a), sizeof(int32_t), cmp_i32);
        ptr[v] = w;
        for (int64_t k = a; k < b; k++) {
            if (k > a && idx[k] == idx[k - 1]) { g->duplicates_dropped++; continue; }
            idx[w++] = idx[k];
        }
    }
    ptr[n] = w;
    g->m = w;
    g->out_ptr = ptr;
    g->out_idx = realloc(idx, (size_t)(w ? w : 1) * sizeof(int32_t));
    if (!g->out_idx) die("out of memory");
}

static void build_in_csr(graph_t *g)
{
    int32_t n = g->n;
    int64_t *ptr = xcalloc((size_t)n + 1, sizeof(int64_t));
    int32_t *idx = xmalloc((size_t)(g->m ? g->m : 1) * sizeof(int32_t));
    for (int64_t e = 0; e < g->m; e++) ptr[g->out_idx[e] + 1]++;
    for (int32_t v = 0; v < n; v++) ptr[v + 1] += ptr[v];
    int64_t *fill = xmalloc((size_t)n * sizeof(int64_t));
    memcpy(fill, ptr, (size_t)n * sizeof(int64_t));
    /* sources visited in ascending order => every in-row is sorted */
    for (int32_t u = 0; u < n; u++)
        for (int64_t k = g->out_ptr[u]; k < g->out_ptr[u + 1]; k++)
            idx[fill[g->out_idx[k]]++] = u;
    free(fill);
    g->in_ptr = ptr;
    g->in_idx = idx;
}

/* Merge two sorted rows without duplicates. If out != NULL, write them. */
static int64_t merge_rows(const int32_t *a, int64_t na, const int32_t *b, int64_t nb, int32_t *out)
{
    int64_t i = 0, j = 0, k = 0;
    while (i < na || j < nb) {
        int32_t x;
        if (j >= nb || (i < na && a[i] < b[j])) x = a[i++];
        else if (i >= na || b[j] < a[i])        x = b[j++];
        else { x = a[i]; i++; j++; }
        if (out) out[k] = x;
        k++;
    }
    return k;
}

static void build_und_csr(graph_t *g)
{
    int32_t n = g->n;
    int64_t *ptr = xcalloc((size_t)n + 1, sizeof(int64_t));
    for (int32_t v = 0; v < n; v++) {
        int64_t len = merge_rows(g->out_idx + g->out_ptr[v], g->out_ptr[v + 1] - g->out_ptr[v],
                                 g->in_idx + g->in_ptr[v], g->in_ptr[v + 1] - g->in_ptr[v], NULL);
        ptr[v + 1] = ptr[v] + len;
    }
    int32_t *idx = xmalloc((size_t)(ptr[n] ? ptr[n] : 1) * sizeof(int32_t));
    for (int32_t v = 0; v < n; v++)
        merge_rows(g->out_idx + g->out_ptr[v], g->out_ptr[v + 1] - g->out_ptr[v],
                   g->in_idx + g->in_ptr[v], g->in_ptr[v + 1] - g->in_ptr[v], idx + ptr[v]);
    if (ptr[n] % 2 != 0) die("internal error: undirected adjacency is not symmetric");
    g->und_ptr = ptr;
    g->und_idx = idx;
    g->m_und = ptr[n] / 2;
}

/* ---------------------------------------------------------------- public */

void graph_load(graph_t *g, const char *mapping_path, const char *edges_path)
{
    memset(g, 0, sizeof(*g));
    load_mapping(g, mapping_path);

    vec_t src = {0}, dst = {0};
    load_edges(g, edges_path, &src, &dst);
    build_out_csr(g, &src, &dst);
    free(src.v);
    free(dst.v);
    build_in_csr(g);
    build_und_csr(g);
}

void graph_free(graph_t *g)
{
    free(g->step);
    free(g->out_ptr); free(g->out_idx);
    free(g->in_ptr);  free(g->in_idx);
    free(g->und_ptr); free(g->und_idx);
    memset(g, 0, sizeof(*g));
}
