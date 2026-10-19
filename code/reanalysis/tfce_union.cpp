// TFCE via component merge trees. Same supplied thresholds and six-neighbor
// components; lazy component scores avoid re-visiting every active voxel.
#include <cmath>
#include <vector>
#include <algorithm>
extern "C" void srtt_tfce(const int n, const double* values,
                          const int* neighbors, const int nt,
                          const double* thresholds, double* result) {
    for (int sign=-1; sign<=1; sign+=2) {
        std::vector<int> order;
        for(int i=0;i<n;++i) if(sign*values[i]>0) order.push_back(i);
        std::sort(order.begin(),order.end(),[&](int a,int b){return sign*values[a]>sign*values[b];});
        std::vector<int> dsu(n,-1), component(n,-1), size(2*n,0), parent(2*n,-1), prev(2*n,-1), next(2*n,-1);
        std::vector<double> score(2*n,0.0);
        int head=-1, nodes=n;
        auto insert=[&](int node){next[node]=head;prev[node]=-1;if(head>=0)prev[head]=node;head=node;};
        auto erase=[&](int node){if(prev[node]>=0)next[prev[node]]=next[node];else head=next[node];if(next[node]>=0)prev[next[node]]=prev[node];};
        auto find=[&](int a){int r=a;while(dsu[r]!=r)r=dsu[r];while(a!=r){int b=dsu[a];dsu[a]=r;a=b;}return r;};
        std::size_t position=0;
        for(int it=nt-1;it>=0;--it){
            const double height=thresholds[it];
            while(position<order.size() && sign*values[order[position]]>=height){
                const int voxel=order[position++];
                dsu[voxel]=voxel;component[voxel]=voxel;size[voxel]=1;insert(voxel);
                for(int d=0;d<6;++d){
                    const int neighbor=neighbors[6*voxel+d];
                    if(neighbor<0 || dsu[neighbor]<0)continue;
                    int a=find(voxel),b=find(neighbor);if(a==b)continue;
                    int ca=component[a],cb=component[b],node=nodes++;
                    size[node]=size[ca]+size[cb];parent[ca]=parent[cb]=node;
                    erase(ca);erase(cb);insert(node);
                    if(size[ca]<size[cb]){dsu[a]=b;component[b]=node;}
                    else{dsu[b]=a;component[a]=node;}
                }
            }
            for(int node=head;node>=0;node=next[node])
                score[node]+=(sign*std::sqrt(static_cast<double>(size[node])))*(height*height);
        }
        for(int node=nodes-1;node>=0;--node)if(parent[node]>=0)score[node]+=score[parent[node]];
        for(int voxel:order)result[voxel]+=score[voxel];
    }
}
