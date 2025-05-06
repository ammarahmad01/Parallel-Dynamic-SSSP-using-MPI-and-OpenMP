// partition_and_save.cpp

// Compile: g++ -O3 -std=c++17 partition_and_save.cpp -lmetis -o partition
// Run:    ./partition <num_partitions>

#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <chrono>
#include <metis.h>

namespace fs = std::filesystem;
using namespace std;
using Clock = std::chrono::high_resolution_clock;

// Build METIS CSR
void buildCSR(int nvtxs,
              const vector<pair<int,int>>& edges,
              vector<idx_t>& xadj,
              vector<idx_t>& adjncy)
{
    xadj.assign(nvtxs+1, 0);
    for (auto &e : edges) {
        xadj[e.first+1]++;
        xadj[e.second+1]++;
    }
    for (int i = 1; i <= nvtxs; ++i)
        xadj[i] += xadj[i-1];

    adjncy.resize(xadj[nvtxs]);
    vector<idx_t> cur = xadj;
    for (auto &e : edges) {
        adjncy[cur[e.first]++] = e.second;
        adjncy[cur[e.second]++] = e.first;
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <num_partitions>\n";
        return 1;
    }
    // Start total timer
    auto t0 = Clock::now();

    int nparts = stoi(argv[1]);
    const string infile = "Dataset/com-lj.ungraph.txt";
    const fs::path outdir = "Parts";

    // 1) Read edges and remap raw IDs
    ifstream fin(infile);
    if (!fin) {
        cerr << "Failed to open " << infile << "\n";
        return 2;
    }
    unordered_map<int,int> idmap;
    vector<int> rev;
    vector<pair<int,int>> edges;
    edges.reserve(10000000);

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
        edges.emplace_back(idmap[u_raw], idmap[v_raw]);
    }
    fin.close();

    int nvtxs = rev.size();

    // 2) Build CSR arrays
    vector<idx_t> xadj, adjncy;
    buildCSR(nvtxs, edges, xadj, adjncy);

    // 3) METIS partition
    vector<idx_t> part(nvtxs);
    idx_t objval, ncon = 1, num_parts = nparts;
    int status = METIS_PartGraphKway(
        &nvtxs, &ncon,
        xadj.data(), adjncy.data(),
        NULL, NULL, NULL,
        &num_parts,
        NULL, NULL, NULL,
        &objval, part.data()
    );
    if (status != METIS_OK) {
        cerr << "METIS error: " << status << "\n";
        return 3;
    }

    // 4) Prepare output directory
    fs::create_directory(outdir);

    // 5) Open one file per part
    vector<ofstream> outs(nparts);
    for (int p = 0; p < nparts; ++p) {
        outs[p].open(outdir / ("part_" + to_string(p) + ".txt"));
        if (!outs[p]) {
            cerr << "Cannot open output for partition " << p << "\n";
            return 4;
        }
    }

    // 6) Write intra-partition edges
    for (auto &e : edges) {
        if (part[e.first] == part[e.second]) {
            int ru = rev[e.first], rv = rev[e.second];
            outs[part[e.first]] << ru << " " << rv << "\n";
        }
    }
    // 7) Close files
    for (auto &o : outs) o.close();

    // Stop total timer
    auto t1 = Clock::now();
    double time_total = chrono::duration<double>(t1 - t0).count();

    cout << "Wrote " << nparts << " files into '" << outdir.string() << "/'.\n"
         << "Time taken (seconds): " << time_total << "\n";

    return 0;
}
