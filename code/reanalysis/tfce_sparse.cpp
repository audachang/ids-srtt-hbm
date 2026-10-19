// Exact threshold-by-threshold TFCE on a sparse six-neighbor voxel graph.
// Thresholds are supplied by Nilearn; no threshold approximation or reordering.
#include <cmath>
#include <vector>
#include <cstdint>
extern "C" void srtt_tfce(const int n, const double* values,
                          const int* neighbors, const int nt,
                          const double* thresholds, double* result) {
    std::vector<int> visited(n, 0), queue(n);
    int stamp = 0;
    for (int sign = -1; sign <= 1; sign += 2) {
        for (int it = 0; it < nt; ++it) {
            ++stamp;
            const double height = thresholds[it];
            for (int start = 0; start < n; ++start) {
                if (visited[start] == stamp || values[start] == 0.0 ||
                    sign * values[start] < height) continue;
                int head = 0, tail = 1;
                queue[0] = start;
                visited[start] = stamp;
                while (head < tail) {
                    const int voxel = queue[head++];
                    for (int d = 0; d < 6; ++d) {
                        const int next = neighbors[voxel * 6 + d];
                        if (next < 0 || visited[next] == stamp ||
                            values[next] == 0.0 || sign * values[next] < height) continue;
                        visited[next] = stamp;
                        queue[tail++] = next;
                    }
                }
                const double contribution = (sign * std::sqrt(static_cast<double>(tail))) * (height * height);
                for (int j = 0; j < tail; ++j) result[queue[j]] += contribution;
            }
        }
    }
}
