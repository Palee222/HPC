#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char mode[16];
    int ranks;
    int threads;
    int cpus;
    double *samples;
    size_t sample_count;
    size_t sample_capacity;
    double median;
    double speedup;
    double efficiency;
} config_t;

static int compare_double(const void *left, const void *right)
{
    double a = *(const double *)left;
    double b = *(const double *)right;
    return (a > b) - (a < b);
}

static config_t *find_or_add(config_t **configs, size_t *count, size_t *capacity,
                             const char *mode, int ranks, int threads, int cpus)
{
    for (size_t i = 0; i < *count; i++) {
        config_t *c = &(*configs)[i];
        if (strcmp(c->mode, mode) == 0 && c->ranks == ranks &&
            c->threads == threads && c->cpus == cpus)
            return c;
    }

    if (*count == *capacity) {
        size_t next = *capacity ? *capacity * 2 : 16;
        config_t *grown = realloc(*configs, next * sizeof(**configs));
        if (!grown) {
            fprintf(stderr, "out of memory storing benchmark configurations\n");
            exit(1);
        }
        *configs = grown;
        *capacity = next;
    }
    config_t *c = &(*configs)[(*count)++];
    memset(c, 0, sizeof(*c));
    snprintf(c->mode, sizeof(c->mode), "%s", mode);
    c->ranks = ranks;
    c->threads = threads;
    c->cpus = cpus;
    return c;
}

static void add_sample(config_t *c, double seconds)
{
    if (c->sample_count == c->sample_capacity) {
        size_t next = c->sample_capacity ? c->sample_capacity * 2 : 4;
        double *grown = realloc(c->samples, next * sizeof(*c->samples));
        if (!grown) {
            fprintf(stderr, "out of memory storing benchmark samples\n");
            exit(1);
        }
        c->samples = grown;
        c->sample_capacity = next;
    }
    c->samples[c->sample_count++] = seconds;
}

static int compare_config(const void *left, const void *right)
{
    const config_t *a = left;
    const config_t *b = right;
    int mode_order = strcmp(a->mode, b->mode);
    if (mode_order) return mode_order;
    if (a->cpus != b->cpus) return (a->cpus > b->cpus) - (a->cpus < b->cpus);
    if (a->ranks != b->ranks) return (a->ranks > b->ranks) - (a->ranks < b->ranks);
    return (a->threads > b->threads) - (a->threads < b->threads);
}

static int same_mode(const config_t *a, const config_t *b)
{
    return strcmp(a->mode, b->mode) == 0;
}

static int write_scaling_csv(const char *path, const config_t *configs, size_t count)
{
    FILE *out = fopen(path, "w");
    if (!out) {
        perror(path);
        return 0;
    }

    fprintf(out, "mode,ranks,threads,total_cpus,repetitions,median_seconds,speedup,efficiency\n");
    for (size_t i = 0; i < count; i++) {
        const config_t *c = &configs[i];
        fprintf(out, "%s,%d,%d,%d,%zu,%.9f,%.9f,%.9f\n",
                c->mode, c->ranks, c->threads, c->cpus, c->sample_count,
                c->median, c->speedup, c->efficiency);
    }
    if (fclose(out) != 0) {
        perror(path);
        return 0;
    }
    return 1;
}

static double chart_x(int cpus, double x0, double width)
{
    return x0 + ((double)cpus - 1.0) * width / 15.0;
}

static double chart_y(double value, double maximum, double y0, double height)
{
    return y0 + height - value * height / maximum;
}

static int write_svg(const char *path, const config_t *configs, size_t count,
                     int efficiency_plot)
{
    static const char *colors[] = {"#2563eb", "#dc2626", "#059669"};
    static const char *modes[] = {"openmp", "mpi", "hybrid"};
    const double x0 = 78.0, y0 = 48.0, width = 770.0, height = 420.0;
    double maximum = efficiency_plot ? 1.1 : 16.0;
    if (efficiency_plot) {
        for (size_t i = 0; i < count; i++)
            if (configs[i].efficiency * 1.1 > maximum)
                maximum = configs[i].efficiency * 1.1;
    } else {
        for (size_t i = 0; i < count; i++)
            if (configs[i].speedup * 1.1 > maximum)
                maximum = configs[i].speedup * 1.1;
    }

    FILE *out = fopen(path, "w");
    if (!out) {
        perror(path);
        return 0;
    }
    fprintf(out,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"920\" height=\"560\" "
        "viewBox=\"0 0 920 560\">\n"
        "<rect width=\"100%%\" height=\"100%%\" fill=\"white\"/>\n"
        "<style>text{font-family:Arial,sans-serif;font-size:13px}"
        ".axis{stroke:#222;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}"
        ".series{fill:none;stroke-width:2.5}</style>\n");
    fprintf(out, "<text x=\"460\" y=\"24\" text-anchor=\"middle\" "
                 "font-size=\"18\">%s</text>\n",
            efficiency_plot ? "Parallel efficiency" : "Speedup");

    int y_ticks = 5;
    for (int i = 0; i <= y_ticks; i++) {
        double value = maximum * i / y_ticks;
        double y = chart_y(value, maximum, y0, height);
        fprintf(out, "<line class=\"grid\" x1=\"%.1f\" y1=\"%.1f\" "
                     "x2=\"%.1f\" y2=\"%.1f\"/>\n", x0, y, x0 + width, y);
        fprintf(out, "<text x=\"%.1f\" y=\"%.1f\" text-anchor=\"end\">%.2f</text>\n",
                x0 - 8, y + 4, value);
    }
    const int cpu_ticks[] = {1, 2, 4, 8, 16};
    for (size_t i = 0; i < sizeof(cpu_ticks) / sizeof(cpu_ticks[0]); i++) {
        double x = chart_x(cpu_ticks[i], x0, width);
        fprintf(out, "<line class=\"grid\" x1=\"%.1f\" y1=\"%.1f\" "
                     "x2=\"%.1f\" y2=\"%.1f\"/>\n", x, y0, x, y0 + height);
        fprintf(out, "<text x=\"%.1f\" y=\"%.1f\" text-anchor=\"middle\">%d</text>\n",
                x, y0 + height + 20, cpu_ticks[i]);
    }
    fprintf(out, "<line class=\"axis\" x1=\"%.1f\" y1=\"%.1f\" "
                 "x2=\"%.1f\" y2=\"%.1f\"/>\n", x0, y0, x0, y0 + height);
    fprintf(out, "<line class=\"axis\" x1=\"%.1f\" y1=\"%.1f\" "
                 "x2=\"%.1f\" y2=\"%.1f\"/>\n",
            x0, y0 + height, x0 + width, y0 + height);
    fprintf(out, "<text x=\"463\" y=\"525\" text-anchor=\"middle\">CPU cores</text>\n");
    fprintf(out, "<text transform=\"translate(18 260) rotate(-90)\" "
                 "text-anchor=\"middle\">%s</text>\n",
            efficiency_plot ? "efficiency" : "speedup");

    if (!efficiency_plot) {
        fprintf(out, "<line x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\" "
                     "stroke=\"#777\" stroke-dasharray=\"6 5\"/>\n",
                chart_x(1, x0, width), chart_y(1, maximum, y0, height),
                chart_x(16, x0, width), chart_y(16, maximum, y0, height));
    } else {
        double y = chart_y(1.0, maximum, y0, height);
        fprintf(out, "<line x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\" "
                     "stroke=\"#777\" stroke-dasharray=\"6 5\"/>\n",
                x0, y, x0 + width, y);
    }

    for (size_t m = 0; m < sizeof(modes) / sizeof(modes[0]); m++) {
        int started = 0;
        for (size_t i = 0; i < count; i++) {
            if (strcmp(configs[i].mode, modes[m]) != 0) continue;
            double value = efficiency_plot ? configs[i].efficiency : configs[i].speedup;
            double x = chart_x(configs[i].cpus, x0, width);
            double y = chart_y(value, maximum, y0, height);
            if (!started) {
                fprintf(out, "<polyline class=\"series\" stroke=\"%s\" points=\"",
                        colors[m]);
                started = 1;
            }
            fprintf(out, "%.1f,%.1f ", x, y);
        }
        if (started) fprintf(out, "\"/>\n");
        for (size_t i = 0; i < count; i++) {
            if (strcmp(configs[i].mode, modes[m]) != 0) continue;
            double value = efficiency_plot ? configs[i].efficiency : configs[i].speedup;
            double x = chart_x(configs[i].cpus, x0, width);
            double y = chart_y(value, maximum, y0, height);
            fprintf(out, "<circle cx=\"%.1f\" cy=\"%.1f\" r=\"4\" fill=\"%s\"/>\n",
                    x, y, colors[m]);
        }
    }

    int legend_y = 500;
    for (size_t m = 0; m < sizeof(modes) / sizeof(modes[0]); m++) {
        int present = 0;
        for (size_t i = 0; i < count; i++)
            if (strcmp(configs[i].mode, modes[m]) == 0) present = 1;
        if (!present) continue;
        fprintf(out, "<line x1=\"%.1f\" y1=\"%d\" x2=\"%.1f\" y2=\"%d\" "
                     "stroke=\"%s\" stroke-width=\"3\"/>\n",
                650.0 + 80.0 * m, legend_y - 4, 670.0 + 80.0 * m,
                legend_y - 4, colors[m]);
        fprintf(out, "<text x=\"%.1f\" y=\"%d\">%s</text>\n",
                675.0 + 80.0 * m, legend_y, modes[m]);
    }
    fprintf(out, "</svg>\n");
    if (fclose(out) != 0) {
        perror(path);
        return 0;
    }
    return 1;
}

int main(int argc, char **argv)
{
    if (argc != 5) {
        fprintf(stderr,
            "usage: speedupplots MEASUREMENTS.csv SCALING.csv "
            "SPEEDUP.svg EFFICIENCY.svg\n");
        return 2;
    }

    FILE *input = fopen(argv[1], "r");
    if (!input) {
        perror(argv[1]);
        return 1;
    }
    char line[1024];
    if (!fgets(line, sizeof(line), input)) {
        fprintf(stderr, "%s is empty\n", argv[1]);
        fclose(input);
        return 1;
    }

    config_t *configs = NULL;
    size_t count = 0, capacity = 0;
    int line_number = 1;
    while (fgets(line, sizeof(line), input)) {
        line_number++;
        char mode[16];
        int ranks, threads, cpus, repetition;
        double seconds;
        if (sscanf(line, "%15[^,],%d,%d,%d,%d,%lf",
                   mode, &ranks, &threads, &cpus, &repetition, &seconds) != 6 ||
            ranks < 1 || threads < 1 || cpus != ranks * threads ||
            repetition < 1 || !(seconds > 0.0)) {
            fprintf(stderr, "%s:%d: invalid benchmark row\n",
                    argv[1], line_number);
            fclose(input);
            free(configs);
            return 1;
        }
        config_t *c = find_or_add(&configs, &count, &capacity,
                                  mode, ranks, threads, cpus);
        add_sample(c, seconds);
    }
    if (fclose(input) != 0) {
        perror(argv[1]);
        free(configs);
        return 1;
    }
    if (count == 0) {
        fprintf(stderr, "%s contains no benchmark rows\n", argv[1]);
        free(configs);
        return 1;
    }

    qsort(configs, count, sizeof(*configs), compare_config);
    for (size_t i = 0; i < count; i++) {
        config_t *c = &configs[i];
        qsort(c->samples, c->sample_count, sizeof(*c->samples), compare_double);
        size_t middle = c->sample_count / 2;
        c->median = c->sample_count % 2
            ? c->samples[middle]
            : (c->samples[middle - 1] + c->samples[middle]) / 2.0;
    }

    for (size_t i = 0; i < count; i++) {
        config_t *c = &configs[i];
        if (i > 0 && same_mode(c, &configs[i - 1])) continue;
        double baseline = 0.0;
        for (size_t j = 0; j < count; j++) {
            if (same_mode(c, &configs[j]) && configs[j].cpus == 1) {
                baseline = configs[j].median;
                break;
            }
        }
        if (!(baseline > 0.0)) {
            fprintf(stderr, "mode '%s' has no 1-CPU baseline\n", c->mode);
            for (size_t j = 0; j < count; j++) free(configs[j].samples);
            free(configs);
            return 1;
        }
        for (size_t j = i; j < count && same_mode(c, &configs[j]); j++) {
            configs[j].speedup = baseline / configs[j].median;
            configs[j].efficiency = configs[j].speedup / configs[j].cpus;
        }
    }

    int ok = write_scaling_csv(argv[2], configs, count) &&
             write_svg(argv[3], configs, count, 0) &&
             write_svg(argv[4], configs, count, 1);
    for (size_t i = 0; i < count; i++) free(configs[i].samples);
    free(configs);
    if (!ok) return 1;
    printf("Wrote %s, %s and %s\n", argv[2], argv[3], argv[4]);
    return 0;
}
