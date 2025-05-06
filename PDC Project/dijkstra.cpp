// sequential dijkstra

// g++ -std=c++17 dijkstra.cpp -o dijkstra
// ./dijkstra


#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <chrono>
#include <algorithm>
#include <numeric>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/strong_components.hpp>

using namespace std;
using namespace std::chrono;

const int INF = numeric_limits<int>::max();

double compute_avg_clustering(const boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS>& g) {
    int n = boost::num_vertices(g);
    double sum = 0.0;
    for (int u = 0; u < n; ++u) {
        vector<int> nbrs;
        for (auto e : boost::make_iterator_range(boost::out_edges(u, g)))
            nbrs.push_back(boost::target(e, g));
        int k = nbrs.size();
        if (k < 2) continue;
        int links = 0;
        for (int i = 0; i < k; ++i) {
            for (int j = i + 1; j < k; ++j) {
                if (boost::edge(nbrs[i], nbrs[j], g).second)
                    ++links;
            }
        }
        sum += (2.0 * links) / (k * (k - 1));
    }
    return sum / n;
}

long long count_triangles(const boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS>& g) {
    int n = boost::num_vertices(g);
    long long tri = 0;
    for (int u = 0; u < n; ++u) {
        vector<int> nbrs;
        for (auto e : boost::make_iterator_range(boost::out_edges(u, g)))
            nbrs.push_back(boost::target(e, g));
        for (size_t i = 0; i < nbrs.size(); ++i) {
            for (size_t j = i + 1; j < nbrs.size(); ++j) {
                if (boost::edge(nbrs[i], nbrs[j], g).second)
                    ++tri;
            }
        }
    }
    return tri / 3; // each triangle counted thrice
}

unordered_map<int, int> dijkstra(const unordered_map<int, vector<int>>& graph, int source) {
    unordered_map<int, int> dist;
    for (const auto& kv : graph) dist[kv.first] = INF;
    dist[source] = 0;
    priority_queue<pair<int,int>, vector<pair<int,int>>, greater<>> pq;
    pq.emplace(0, source);
    while (!pq.empty()) {
        auto [d,u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        for (int v : graph.at(u)) {
            if (dist[v] > d + 1) {
                dist[v] = d + 1;
                pq.emplace(dist[v], v);
            }
        }
    }
    return dist;
}

int main() {
    auto t0 = high_resolution_clock::now();

    unordered_map<int, vector<int>> graph;
    string line;
    ifstream infile("Dataset/com-amazon.ungraph.txt");
    if (!infile.is_open()) {
        cerr << "Failed to open graph file." << endl;
        return 1;
    }
    while (getline(infile, line)) {
        if (line.empty() || line[0] == '#') continue;
        istringstream iss(line);
        int u, v;
        if (!(iss >> u >> v)) continue;
        graph[u].push_back(v);
        graph[v].push_back(u);
    }
    infile.close();

    // Map node IDs to contiguous indices for Boost
    unordered_map<int,int> idmap;
    vector<int> rev;
    rev.reserve(graph.size());
    for (auto& kv : graph) {
        int node = kv.first;
        if (!idmap.count(node)) {
            idmap[node] = rev.size();
            rev.push_back(node);
        }
    }
    int N = rev.size();
    // Build Boost graph
    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> BGraph;
    BGraph bg(N);
    long long edge_count = 0;
    for (auto& kv : graph) {
        int u = idmap[kv.first];
        for (int v_raw : kv.second) {
            int v = idmap[v_raw];
            if (u < v) {
                boost::add_edge(u, v, bg);
                ++edge_count;
            }
        }
    }

    // Compute WCC
    vector<int> comp(N);
    int wcc = boost::connected_components(bg, comp.data());
    vector<int> csize(wcc, 0);
    for (int cid : comp) csize[cid]++;
    int lw = max_element(csize.begin(), csize.end()) - csize.begin();
    int nodes_lwcc = csize[lw];
    int edges_lwcc = 0;
    for (auto e : boost::make_iterator_range(boost::edges(bg))) {
        int su = boost::source(e,bg), tv = boost::target(e,bg);
        if (comp[su]==lw && comp[tv]==lw) ++edges_lwcc;
    }
    // SCC on undirected == WCC
    int nodes_lscc = nodes_lwcc;
    int edges_lscc = edges_lwcc;

    double avg_clust = compute_avg_clustering(bg);
    long long triangles = count_triangles(bg);

    auto t1 = high_resolution_clock::now();
    double time_taken = duration_cast<duration<double>>(t1 - t0).count();

    // Output
    cout << "Nodes\t" << N << "\n"
         << "Edges\t" << edge_count << "\n"
         << "Nodes in largest WCC\t" << nodes_lwcc << "\n"
         << "Edges in largest WCC\t" << edges_lwcc << "\n"
         << "Nodes in largest SCC\t" << nodes_lscc << "\n"
         << "Edges in largest SCC\t" << edges_lscc << "\n"
         << "Average clustering coefficient\t" << avg_clust << "\n"
         << "Number of triangles\t" << triangles << "\n"
         << "\nTime Taken\t" << time_taken << endl;

    return 0;
}
