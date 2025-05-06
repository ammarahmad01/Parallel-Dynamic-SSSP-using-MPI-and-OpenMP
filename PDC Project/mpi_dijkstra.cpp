#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <algorithm>
#include <sstream>
#include <numeric>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/strong_components.hpp>

using std::cout;
using std::endl;

// Graph types
typedef boost::adjacency_list<
    boost::vecS,
    boost::vecS,
    boost::undirectedS,
    boost::no_property,
    boost::property<boost::edge_weight_t, int>
> Graph;

static const int INF = std::numeric_limits<int>::max();
typedef std::unordered_map<int,int> DistanceMap;

double compute_avg_clustering(const Graph& g) {
    int n = boost::num_vertices(g);
    double sum = 0.0;
    for (int u = 0; u < n; ++u) {
        std::vector<int> nbrs;
        for (auto e : boost::make_iterator_range(boost::out_edges(u, g)))
            nbrs.push_back(boost::target(e, g));
        int k = nbrs.size();
        if (k < 2) continue;
        int links = 0;
        for (int i = 0; i < k; ++i) {
            for (int j = i+1; j < k; ++j) {
                if (boost::edge(nbrs[i], nbrs[j], g).second)
                    ++links;
            }
        }
        sum += (2.0 * links) / (k * (k - 1));
    }
    return sum / n;
}

long long count_triangles(const Graph& g) {
    int n = boost::num_vertices(g);
    long long tri = 0;
    for (int u = 0; u < n; ++u) {
        std::vector<int> nbrs;
        for (auto e : boost::make_iterator_range(boost::out_edges(u, g)))
            nbrs.push_back(boost::target(e, g));
        for (int i = 0; i < (int)nbrs.size(); ++i) {
            for (int j = i+1; j < (int)nbrs.size(); ++j) {
                if (boost::edge(nbrs[i], nbrs[j], g).second)
                    ++tri;
            }
        }
    }
    return tri / 3; // each triangle counted at each vertex
}

DistanceMap dijkstra_local(const Graph& g, int source) {
    DistanceMap dist;
    std::unordered_set<int> visited;
    std::priority_queue<std::pair<int,int>, std::vector<std::pair<int,int>>, std::greater<>> pq;

    for (auto vp = boost::vertices(g); vp.first != vp.second; ++vp.first)
        dist[*vp.first] = INF;
    dist[source] = 0;
    pq.emplace(0, source);

    while (!pq.empty()) {
        auto [d,u] = pq.top(); pq.pop();
        if (!visited.insert(u).second) continue;
        for (auto ep = boost::out_edges(u, g); ep.first != ep.second; ++ep.first) {
            int v = boost::target(*ep.first, g);
            int w = boost::get(boost::edge_weight, g, *ep.first);
            if (dist[v] > d + w) {
                dist[v] = d + w;
                pq.emplace(dist[v], v);
            }
        }
    }
    return dist;
}

Graph load_graph(const std::string& filename, int& node_count) {
    Graph g;
    std::unordered_map<int,int> idmap;
    node_count = 0;
    std::ifstream infile(filename);
    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        int u_raw, v_raw;
        ss >> u_raw >> v_raw;
        if (!idmap.count(u_raw)) idmap[u_raw] = node_count++;
        if (!idmap.count(v_raw)) idmap[v_raw] = node_count++;
        int u = idmap[u_raw], v = idmap[v_raw];
        boost::add_edge(u, v, 1, g);
    }
    return g;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double t0 = MPI_Wtime();

    Graph g;
    int total_nodes=0, total_edges=0;
    if (rank == 0) {
        g = load_graph("Dataset/com-youtube.ungraph.txt", total_nodes);
        total_edges = boost::num_edges(g);
    }
    MPI_Bcast(&total_nodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&total_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast edges
    std::vector<std::pair<int,int>> edges;
    if (rank==0) {
        edges.reserve(total_edges);
        for (auto e: boost::make_iterator_range(boost::edges(g)))
            edges.emplace_back(boost::source(e,g), boost::target(e,g));
    }
    int ec = edges.size();
    MPI_Bcast(&ec, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<int> buf(2*ec);
    if (rank==0) for (int i=0;i<ec;++i) { buf[2*i]=edges[i].first; buf[2*i+1]=edges[i].second; }
    MPI_Bcast(buf.data(), 2*ec, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank!=0) {
        g = Graph(total_nodes);
        for (int i=0;i<ec;++i)
            boost::add_edge(buf[2*i], buf[2*i+1], 1, g);
    }

    if (rank==0) {
        // WCC
        std::vector<int> comp(total_nodes);
        int wcc = boost::connected_components(g, &comp[0]);
        std::vector<int> csize(wcc);
        for (int v: comp) csize[v]++;
        int largest_wcc = *std::max_element(csize.begin(), csize.end());
        // edges in largest WCC
        std::vector<bool> in_lwcc(total_nodes);
        int lw = std::distance(csize.begin(), std::max_element(csize.begin(), csize.end()));
        for (int i=0;i<total_nodes;++i) if (comp[i]==lw) in_lwcc[i]=true;
        int edges_lwcc=0;
        for (auto e: boost::make_iterator_range(boost::edges(g)))
            if (in_lwcc[boost::source(e,g)] && in_lwcc[boost::target(e,g)])
                edges_lwcc++;
        // SCC on undirected == WCC
        int largest_scc = largest_wcc;
        int edges_scc = edges_lwcc;
        // clustering & triangles
        double avg_clust = compute_avg_clustering(g);
        long long triangles = count_triangles(g);
        double t1 = MPI_Wtime();

        cout<<"Nodes\t"<<total_nodes<<"\n"
            <<"Edges\t"<<total_edges<<"\n"
            <<"Nodes in largest WCC\t"<<largest_wcc<<"\n"
            <<"Edges in largest WCC\t"<<edges_lwcc<<"\n"
            <<"Nodes in largest SCC\t"<<largest_scc<<"\n"
            <<"Edges in largest SCC\t"<<edges_scc<<"\n"
            <<"Average clustering coefficient\t"<<avg_clust<<"\n"
            <<"Number of triangles\t"<<triangles<<"\n"
            <<"\nTime Taken\t"<<(t1-t0)<<"\n";
    }

    MPI_Finalize();
    return 0;
}
