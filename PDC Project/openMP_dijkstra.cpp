// openmp_dijkstra_with_metrics.cpp

// Compile: g++ -O3 -std=c++17 openmp_dijkstra_with_metrics.cpp -fopenmp -lmetis -o openmp

// Run:
//   export OMP_NUM_THREADS=4
//   ./openmp

#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <omp.h>
#include <metis.h>

using namespace std;
namespace fs = std::filesystem;
static const int INF = numeric_limits<int>::max();

// Dijkstra on adjacency-list
unordered_map<int,int> dijkstra_local(const vector<vector<int>>& adj, int source) {
    unordered_map<int,int> dist;
    dist.reserve(adj.size());
    for (int u = 0; u < (int)adj.size(); ++u) dist[u] = INF;
    dist[source] = 0;

    priority_queue<pair<int,int>, vector<pair<int,int>>, greater<>> pq;
    pq.emplace(0, source);

    while (!pq.empty()) {
        auto [d,u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        for (int v : adj[u]) {
            if (dist[v] > d + 1) {
                dist[v] = d + 1;
                pq.emplace(dist[v], v);
            }
        }
    }
    return dist;
}

// BFS for WCC
pair<vector<int>,vector<int>> compute_WCC(const vector<vector<int>>& adj) {
    int N = adj.size();
    vector<char> vis(N, 0);
    vector<int> sizes, edges;
    sizes.reserve(N);
    edges.reserve(N);

    for (int i = 0; i < N; ++i) {
        if (vis[i]) continue;
        queue<int> q; q.push(i);
        vis[i] = 1;
        int cnt = 0, ecnt = 0;
        while (!q.empty()) {
            int u = q.front(); q.pop();
            cnt++; 
            ecnt += adj[u].size();
            for (int v : adj[u]) {
                if (!vis[v]) {
                    vis[v] = 1;
                    q.push(v);
                }
            }
        }
        sizes.push_back(cnt);
        edges.push_back(ecnt / 2);
    }
    return {sizes, edges};
}

// Clustering & triangles
pair<double,long long> clustering_and_triangles(const vector<vector<int>>& adj) {
    int N = adj.size();
    vector<unordered_set<int>> neigh(N);
    for (int u = 0; u < N; ++u)
        for (int v : adj[u])
            neigh[u].insert(v);

    double cc_sum = 0.0;
    long long tri = 0;

    for (int u = 0; u < N; ++u) {
        int d = adj[u].size();
        if (d < 2) continue;
        int links = 0;
        for (int i = 0; i < d; ++i) {
            for (int j = i + 1; j < d; ++j) {
                if (neigh[adj[u][i]].count(adj[u][j])) links++;
            }
        }
        cc_sum += (2.0 * links) / (d * (d - 1));
    }

    for (int u = 0; u < N; ++u) {
        for (int v : adj[u]) if (v > u) {
            for (int w : adj[v]) if (w > v) {
                if (neigh[u].count(w)) tri++;
            }
        }
    }

    return {cc_sum / N, tri};
}

int main() {
    // ─── Start total timer ─────────────────────────────────────
    double t0 = omp_get_wtime();

    // 1) Load & remap graph
    const string infile = "Dataset/com-lj.ungraph.txt";
    ifstream fin(infile);
    if (!fin) { 
        cerr << "Cannot open " << infile << "\n"; 
        return 1; 
    }

    unordered_map<int,int> idmap;
    vector<int> rev;
    vector<pair<int,int>> edges_raw;
    edges_raw.reserve(5000000);

    int u_raw, v_raw;
    while (fin >> u_raw >> v_raw) {
        if (!idmap.count(u_raw)) { 
            idmap[u_raw] = rev.size(); 
            rev.push_back(u_raw); 
        }
        if (!idmap.count(v_raw)) { 
            idmap[v_raw] = rev.size(); 
            rev.push_back(v_raw); 
        }
        edges_raw.emplace_back(idmap[u_raw], idmap[v_raw]);
    }
    fin.close();

    int N = rev.size();
    vector<vector<int>> adj(N);
    for (auto &e : edges_raw) {
        adj[e.first].push_back(e.second);
        adj[e.second].push_back(e.first);
    }

    // 2) Compute global metrics
    int total_nodes = N;
    long long total_edges = edges_raw.size();

    auto [wcc_sz, wcc_ed] = compute_WCC(adj);
    int idx = max_element(wcc_sz.begin(), wcc_sz.end()) - wcc_sz.begin();
    int nodes_lwcc = wcc_sz[idx], edges_lwcc = wcc_ed[idx];

    // SCC ≡ WCC for undirected
    int nodes_lscc = nodes_lwcc, edges_lscc = edges_lwcc;

    auto [avg_cc, tri_count] = clustering_and_triangles(adj);

    // 3) Discover partitions in Parts/
    vector<pair<int,fs::path>> parts;
    for (auto &ent : fs::directory_iterator("Parts")) {
        auto fn = ent.path().filename().string();
        if (ent.path().extension()==".txt" && fn.rfind("part_",0)==0) {
            int p = stoi(fn.substr(5, fn.find('.')-5));
            parts.emplace_back(p, ent.path());
        }
    }
    sort(parts.begin(), parts.end());
    int T = parts.size();

    // 4) Read each partition file
    vector<int> part_of(N, -1);
    for (auto &pp : parts) {
        int pid = pp.first;
        ifstream pf(pp.second);
        while (pf >> u_raw >> v_raw) {
            part_of[idmap[u_raw]] = pid;
            part_of[idmap[v_raw]] = pid;
        }
    }

    vector<vector<int>> nodes_per(T);
    for (int u = 0; u < N; ++u)
        if (part_of[u] >= 0)
            nodes_per[part_of[u]].push_back(u);

    // 5) Parallel Dijkstra
    vector<unordered_map<int,int>> results(T);
    #pragma omp parallel num_threads(T)
    {
        int tid = omp_get_thread_num();
        vector<vector<int>> subadj(N);
        for (int u : nodes_per[tid]) {
            for (int v : adj[u]) {
                if (part_of[v] == tid)
                    subadj[u].push_back(v);
            }
        }
        int source = nodes_per[0][0];
        if (find(nodes_per[tid].begin(), nodes_per[tid].end(), source)
            != nodes_per[tid].end())
        {
            results[tid] = dijkstra_local(subadj, source);
        }
    }

    // ─── End total timer ───────────────────────────────────────
    double t1 = omp_get_wtime();
    double time_total = t1 - t0;

    // ─── Print exactly the requested metrics ───────────────────
    cout
      << "Nodes                 : " << total_nodes   << "\n"
      << "Edges                 : " << total_edges   << "\n"
      << "Nodes in largest WCC  : " << nodes_lwcc    << "\n"
      << "Edges in largest WCC  : " << edges_lwcc    << "\n"
      << "Nodes in largest SCC  : " << nodes_lscc    << "\n"
      << "Edges in largest SCC  : " << edges_lscc    << "\n"
      << "Average clustering coeff. : " << avg_cc     << "\n"
      << "Number of triangles   : " << tri_count     << "\n"
      << "\nTime taken (seconds)  : " << time_total    << "\n";

    return 0;
}
