
// #define ORIG
// #define LINEAR
// #define POWER_LAW
// #define ADDITIVE_BUMP
// #define CAPPED_MULT_THRESH

double TOT_RSMT_LENGTH = 0;
vector<vector<int>> my_flute(unordered_set<int> &pos) {
    const int MAX_DEGREE = 10000;
    vector<int> x;
    vector<int> y;
    int cnt = 0;
    vector<int> nodes, parent;
    vector<tuple<int, int, int>> edges;

#ifdef ORIG

    for(auto e : pos) {
        x.push_back(db::dr_x[e / Y]);
        y.push_back(db::dr_y[e % Y]);
        cnt++;
    }
    auto tree = flute::flute(cnt, x.data(), y.data(), 3);
    for(int i = 0; i < cnt * 2 - 2; i++) {
        flute::Branch &branch = tree.branch[i];
        nodes.emplace_back(db::dr2gr_x[branch.x] * Y + db::dr2gr_y[branch.y]);
    }    
    sort(nodes.begin(), nodes.end());
    nodes.erase(unique(nodes.begin(), nodes.end()), nodes.end());
    parent.resize(nodes.size());
    for(int i = 0; i < nodes.size(); i++) parent[i] = i;
    edges.reserve(cnt * 2);
    for(int i = 0; i < cnt * 2 - 2; i++) if(tree.branch[i].n < cnt * 2 - 2) {
        Branch &branch1 = tree.branch[i], &branch2 = tree.branch[branch1.n];
        int u, v;
        u = lower_bound(nodes.begin(), nodes.end(), db::dr2gr_x[branch1.x] * Y + db::dr2gr_y[branch1.y]) - nodes.begin();
        v = lower_bound(nodes.begin(), nodes.end(), db::dr2gr_x[branch2.x] * Y + db::dr2gr_y[branch2.y]) - nodes.begin();
        if(u == v) continue;
        edges.emplace_back(make_tuple(abs(branch1.x - branch2.x) + abs(branch1.y - branch2.y), u, v));
            
    }
    sort(edges.begin(), edges.end());
    function<int(int)> find_parent = [&] (int x) { return x == parent[x] ? x : parent[x] = find_parent(parent[x]); };
    vector<vector<int>> graph(nodes.size());
    for(auto edge : edges) {
        int u = get<1> (edge), v = get<2> (edge), par_u = find_parent(u), par_v = find_parent(v);
        if(par_u != par_v) {
            graph[u].emplace_back(v);
            graph[v].emplace_back(u);
            TOT_RSMT_LENGTH += get<0> (edge);
            parent[par_u] = par_v;
        }
    }
    int tot_degree = 0;
    for(int i = 0; i < nodes.size(); i++) tot_degree += graph[i].size();
    assert(tot_degree == 2 * (nodes.size() - 1));
    graph.emplace_back(move(nodes));
    return move(graph);

#else

    for(auto e : pos) {
	int dbx = db::dr_x[e / Y];
	int dby = db::dr_y[e % Y];
	int grx = db::dr2gr_x[dbx];
	int gry = db::dr2gr_y[dby];
        x.push_back(grx);
        y.push_back(gry);
	cnt++;
    }

    vector<int> xs(cnt), ys(cnt), s(cnt);

    vector<int> idx(cnt);
    iota(idx.begin(), idx.end(), 0);

    vector<int> idx_x = idx;
    sort(idx_x.begin(), idx_x.end(),
        [&](int a, int b) { return x[a] < x[b]; });

    vector<int> x_rank(cnt);
    for (int i = 0; i < cnt; ++i) {
        xs[i] = x[idx_x[i]];
        x_rank[idx_x[i]] = i;
    }

    vector<int> idx_y = idx;
    sort(idx_y.begin(), idx_y.end(),
        [&](int a, int b) { return y[a] < y[b]; });

    for (int i = 0; i < cnt; ++i) {
        ys[i] = y[idx_y[i]];
        s[i] = x_rank[idx_y[i]];
    }

    vector<int> x_seg(cnt - 1), y_seg(cnt - 1);
    for (int i = 0; i < cnt - 1; i++) {
        x_seg[i] = (xs[i + 1] - xs[i]) * 100;
        y_seg[i] = (ys[i + 1] - ys[i]) * 100;
    }

#ifdef LINEAR

    std::vector<int> orig_x_seg = x_seg;
    std::vector<int> orig_y_seg = y_seg;
    
    int M = cnt - 1;
    std::vector<int>  x_sum(M), y_sum(M), x_raw_warp(M), y_raw_warp(M);
    std::vector<float> x_avg(M), x_norm(M), y_avg(M), y_norm(M);

    float coeffH = 5.75f;
    float coeffV = 2.1f;

    for(int i = 0; i < M; ++i) {
        int sum = 0;
        for(int yy = 0; yy < Y; ++yy)
          for(int xx = xs[i]; xx < xs[i+1]; ++xx)
            sum += congestion_matrix[xx + yy * X];
    
        x_sum[i]   = sum;
        x_avg[i]   = float(sum) / ((xs[i+1] - xs[i]) * Y);
        x_norm[i]  = x_avg[i] / 255.0f;
    
        int raw = int(std::round(orig_x_seg[i] * (1.0f + coeffH * x_norm[i])));
        x_raw_warp[i] = raw;
        x_seg[i]      = std::max(1, raw);
    }
    
    for(int i = 0; i < M; ++i) {
        int sum = 0;
        for(int xx = 0; xx < X; ++xx)
          for(int yy = ys[i]; yy < ys[i+1]; ++yy)
            sum += congestion_matrix[xx + yy * X];
    
        y_sum[i]   = sum;
        y_avg[i]   = float(sum) / ((ys[i+1] - ys[i]) * X);
        y_norm[i]  = y_avg[i] / 255.0f;
    
        int raw = int(std::round(orig_y_seg[i] * (1.0f + coeffV * y_norm[i])));
        y_raw_warp[i] = raw;
        y_seg[i]      = std::max(1, raw);
    }

    // static int debug_run = 0;
    // if (debug_run < 10) {
    //     std::ofstream dbg("warp_debug.txt", std::ios::app);
    //     dbg << "=== Invocation " << debug_run++ << " ===\n";
    // 
    //     for(int i = 0; i < M; ++i) {
    //         dbg << "Segment " << i << " (H): "
    //             << "orig=" << orig_x_seg[i]
    //             << ", sum="   << x_sum[i]
    //             << ", avg="   << x_avg[i]
    //             << ", norm="  << x_norm[i]
    //             << ", formula=round(" << orig_x_seg[i]
    //                << "*(1+" << coeffH << "*" << x_norm[i] << "))"
    //             << "="     << x_raw_warp[i]
    //             << " -> final=" << x_seg[i]
    //             << "\n";
    // 
    //         dbg << "Segment " << i << " (V): "
    //             << "orig=" << orig_y_seg[i]
    //             << ", sum="   << y_sum[i]
    //             << ", avg="   << y_avg[i]
    //             << ", norm="  << y_norm[i]
    //             << ", formula=round(" << orig_y_seg[i]
    //                << "*(1+" << coeffV << "*" << y_norm[i] << "))"
    //             << "="     << y_raw_warp[i]
    //             << " -> final=" << y_seg[i]
    //             << "\n";
    //     }
    //     dbg << "\n";
    // }

#elif defined(POWER_LAW)

    std::vector<int> orig_x_seg = x_seg;
    std::vector<int> orig_y_seg = y_seg;

    int M = cnt - 1;
    
    float coeffH = 5.75f, coeffV = 2.1f;
    float pH = 1000.0f,  pV = 1000.0f;

    for(int i = 0; i < M; ++i) {
        int sum = 0;
        for(int yy = 0; yy < Y; ++yy)
          for(int xx = xs[i]; xx < xs[i+1]; ++xx)
            sum += congestion_matrix[xx + yy * X];
    
        float avg  = float(sum) / ((xs[i+1] - xs[i]) * Y);
        float norm = (avg/255.0f);
        float normp = powf(norm, pH);
    
        int raw = int(std::round(orig_x_seg[i] * (1.0f + coeffH * normp)));
        x_seg[i] = std::max(1, raw);
    }
    
    for(int i = 0; i < M; ++i) {
        int sum = 0;
        for(int xx = 0; xx < X; ++xx)
          for(int yy = ys[i]; yy < ys[i+1]; ++yy)
            sum += congestion_matrix[xx + yy * X];
    
        float avg  = float(sum) / ((ys[i+1] - ys[i]) * X);
        float norm = (avg/255.0f);
        float normp = powf(norm, pV);
    
        int raw = int(std::round(orig_y_seg[i] * (1.0f + coeffV * normp)));
        y_seg[i] = std::max(1, raw);
    }

#elif defined(ADDITIVE_BUMP)

    std::vector<int> orig_x_seg = x_seg;
    std::vector<int> orig_y_seg = y_seg;

    int M = cnt - 1;
    
    float coeffH = 5.1f, coeffV = 7.2f;
    int minBump = -1;
    
    for(int i = 0; i < M; ++i) {
        int sum = 0;
        for(int yy = 0; yy < Y; ++yy)
          for(int xx = xs[i]; xx < xs[i+1]; ++xx)
            sum += congestion_matrix[xx + yy * X];
    
        float avg = float(sum) / ((xs[i+1] - xs[i]) * Y);
        int bump = (sum > 0 ? minBump : 0);
        int raw  = orig_x_seg[i] + int(coeffH * avg) + bump;
        x_seg[i] = std::max(1, raw);
    }
    
    for(int i = 0; i < M; ++i) {
        int sum = 0;
        for(int xx = 0; xx < X; ++xx)
          for(int yy = ys[i]; yy < ys[i+1]; ++yy)
            sum += congestion_matrix[xx + yy * X];
    
        float avg = float(sum) / ((ys[i+1] - ys[i]) * X);
        int bump = (sum > 0 ? minBump : 0);
        int raw  = orig_y_seg[i] + int(coeffV * avg) + bump;
        y_seg[i] = std::max(1, raw);
    }

#elif defined(EXPONENTIAL_WARP)

    std::vector<int> orig_x_seg = x_seg;
    std::vector<int> orig_y_seg = y_seg;
    int M = cnt - 1;

    float coeffH = 3.0f, coeffV = 1.5f;

    for(int i = 0; i < M; ++i) {
        // horizontal
       int sum = 0;
        for(int yy = 0; yy < Y; ++yy)
          for(int xx = xs[i]; xx < xs[i+1]; ++xx)
            sum += congestion_matrix[xx + yy * X];
            
        float avg = float(sum)/((xs[i+1]-xs[i]) * Y);
        float norm = avg/255.0f;
        float warp = expf(coeffH * norm);
        x_seg[i] = std::max(1, int(std::round(orig_x_seg[i] * warp)));
    }

    for(int i = 0; i < M; ++i) {
        // vertical
        int sum = 0;
        for(int xx = 0; xx < X; ++xx)
          for(int yy = ys[i]; yy < ys[i+1]; ++yy)
            sum += congestion_matrix[xx + yy * X];

        float avg = float(sum)/((ys[i+1]-ys[i]) * X);
        float norm = avg/255.0f;
        float warp = expf(coeffV * norm);
        y_seg[i] = std::max(1, int(std::round(orig_y_seg[i] * warp)));
    }
//#define GAUSSIAN_WARP
...
#elif defined(GAUSSIAN_WARP)

    // Idea: Use a Gaussian bump on normalized congestion to emphasize mid-range hot spots.
    // warp = 1 + coeff * exp( -((norm - μ)^2) / (2σ^2) )
  
    std::vector<int> orig_x_seg = x_seg;
    std::vector<int> orig_y_seg = y_seg;
    int M = cnt - 1;

    float coeffH = 5.0f, coeffV = 2.0f;
    float mu     = 0.5f, sigma = 0.2f;

    for(int i = 0; i < M; ++i) {
        // horizontal
        int sum = 0;
        for(int yy = 0; yy < Y; ++yy)
          for(int xx = xs[i]; xx < xs[i+1]; ++xx)
            sum += congestion_matrix[xx + yy * X];

        float norm = (float(sum) / ((xs[i+1]-xs[i]) * Y)) / 255.0f;
        float gauss = expf(-((norm - mu)*(norm - mu)) / (2.0f * sigma * sigma));
        float warp  = 1.0f + coeffH * gauss;

        x_seg[i] = std::max(1, int(std::round(orig_x_seg[i] * warp)));
    }

    for(int i = 0; i < M; ++i) {
        // vertical
        int sum = 0;
        for(int xx = 0; xx < X; ++xx)
          for(int yy = ys[i]; yy < ys[i+1]; ++yy)
            sum += congestion_matrix[xx + yy * X];

        float norm = (float(sum) / ((ys[i+1]-ys[i]) * X)) / 255.0f;
        float gauss = expf(-((norm - mu)*(norm - mu)) / (2.0f * sigma * sigma));
        float warp  = 1.0f + coeffV * gauss;

        y_seg[i] = std::max(1, int(std::round(orig_y_seg[i] * warp)));
    }

#else
    #error "Implementation must be defined"

#endif
