#include <stdio.h> 
#include <stdlib.h> 
#include <limits.h> 
#include <time.h> 
  
#define MAXV  105 
#define INF   INT_MAX 
  
/* ── Adjacency list structures ── */ 
typedef struct Edge { int to, w, next; } Edge; 
Edge edgeList[MAXV * MAXV]; 
int  head[MAXV], edgeCnt; 
  
void initGraph(int V) { 
    edgeCnt = 0; 
    for (int i = 0; i < V; i++) head[i] = -1; 
} 
  
void addEdge(int u, int v, int w) { 
    edgeList[edgeCnt] = (Edge){v, w, head[u]}; 
    head[u] = edgeCnt++; 
} 
  
/* ── Min-heap for Dijkstra ── */ 
typedef struct { int dist, v; } Node; 
Node heap[MAXV * MAXV]; 
int  heapSz; 
  
void heapPush(int d, int v) { 
    int i = ++heapSz; 
    heap[i] = (Node){d, v}; 
    while (i > 1 && heap[i].dist < heap[i/2].dist) { 
        Node tmp = heap[i]; heap[i] = heap[i/2]; heap[i/2] = tmp; 
        i /= 2; 
    } 
} 
  
Node heapPop(void) { 
    Node top = heap[1]; 
    heap[1]  = heap[heapSz--]; 
    int i = 1; 
    while (1) { 
        int s = i, l = 2*i, r = 2*i+1; 
        if (l <= heapSz && heap[l].dist < heap[s].dist) s = l; 
        if (r <= heapSz && heap[r].dist < heap[s].dist) s = r; 
        if (s == i) break; 
        Node tmp = heap[i]; heap[i] = heap[s]; heap[s] = tmp; 
        i = s; 
    } 
    return top; 
} 
  
/* ═══════════════════════════════════════════════ 
   DIJKSTRA  —  O((V + E) log V) 
   ═══════════════════════════════════════════════ */ 
void dijkstra(int src, int V, long long dist[]) { 
    for (int i = 0; i < V; i++) dist[i] = INF; 
    dist[src] = 0; 
    heapSz = 0; 
    heapPush(0, src); 
  
    while (heapSz > 0) { 
        Node cur = heapPop(); 
        int u = cur.v, d = cur.dist; 
        if (d > dist[u]) continue;           /* stale entry */ 
        for (int e = head[u]; e != -1; e = edgeList[e].next) { 
            int v = edgeList[e].to; 
            int w = edgeList[e].w; 
            if (dist[u] + w < dist[v]) { 
                dist[v] = dist[u] + w; 
                heapPush((int)dist[v], v); 
            } 
        } 
    } 
} 
  
/* ═══════════════════════════════════════════════ 
   BELLMAN-FORD  —  O(V * E) 
   Detects negative weight cycles. 
   ═══════════════════════════════════════════════ */ 
typedef struct { int u, v, w; } BFEdge; 
BFEdge bfEdges[MAXV * MAXV]; 
int    bfEdgeCnt; 
  
void buildBFEdges(int V) { 
    bfEdgeCnt = 0; 
    for (int u = 0; u < V; u++) 
        for (int e = head[u]; e != -1; e = edgeList[e].next) 
            bfEdges[bfEdgeCnt++] = (BFEdge){u, edgeList[e].to, edgeList[e].w}; 
} 
  
int bellmanFord(int src, int V, long long dist[]) { 
    for (int i = 0; i < V; i++) dist[i] = INF; 
    dist[src] = 0; 
  
    for (int iter = 0; iter < V - 1; iter++) 
        for (int i = 0; i < bfEdgeCnt; i++) { 
            int u=bfEdges[i].u, v=bfEdges[i].v, w=bfEdges[i].w; 
            if (dist[u] != INF && dist[u] + w < dist[v]) 
                dist[v] = dist[u] + w; 
        } 
  
    /* Negative cycle detection — V-th relaxation */ 
    for (int i = 0; i < bfEdgeCnt; i++) { 
        int u=bfEdges[i].u, v=bfEdges[i].v, w=bfEdges[i].w; 
        if (dist[u] != INF && dist[u] + w < dist[v]) 
            return 1;   /* negative cycle exists */ 
    } 
    return 0; 
} 
  
void printDist(long long dist[], int V, const char *algo) { 
    printf("%s distances: ", algo); 
    for (int i = 0; i < V; i++) { 
        if (dist[i] == INF) printf("INF "); 
        else                printf("%lld ", dist[i]); 
    } 
    printf("\n"); 
} 
  
double getTimeMs(void) { 
    struct timespec ts; 
    clock_gettime(CLOCK_MONOTONIC, &ts); 
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6; 
} 
  
int main(void) { 
  
    /* ── MANDATORY TEST (V=5, source=0) ── */ 
    printf("=== Mandatory Test: V=5, Source=0 ===\n"); 
    int V0 = 5; 
    initGraph(V0); 
    addEdge(0,1,4); addEdge(0,2,2); 
    addEdge(1,2,1); addEdge(1,3,5); 
    addEdge(2,3,8); addEdge(2,4,10); 
    addEdge(3,4,2); 
    buildBFEdges(V0); 
  
    long long dD[MAXV], dBF[MAXV]; 
    double t0, t1; 
  
    t0 = getTimeMs(); dijkstra(0, V0, dD);  t1 = getTimeMs(); 
    printf("Dijkstra     : "); printDist(dD, V0, ""); 
    printf("Dijkstra time: %.4f ms\n", t1-t0); 
  
    t0 = getTimeMs(); 
    int neg = bellmanFord(0, V0, dBF); t1 = getTimeMs(); 
    printf("Bellman-Ford : "); printDist(dBF, V0, ""); 
    printf("BF time      : %.4f ms\n", t1-t0); 
    printf("Negative cycle: %s\n", neg ? "DETECTED" : "None"); 
  
    /* ── NEGATIVE EDGE CHALLENGE ── */ 
    printf("\n=== Negative Edge Challenge ===\n"); 
    int Vn = 5; initGraph(Vn); 
    addEdge(0,1,4); addEdge(0,2,2); addEdge(1,3,5); 
    addEdge(2,1,-1); addEdge(2,3,8); addEdge(3,4,2); 
    buildBFEdges(Vn); 
    neg = bellmanFord(0, Vn, dBF); 
    printDist(dBF, Vn, "Bellman-Ford"); 
    printf("Negative cycle: %s\n", neg ? "DETECTED" : "None"); 
  
    /* ── NEGATIVE CYCLE CHALLENGE ── */ 
    printf("\n=== Negative Cycle Challenge ===\n"); 
int Vc = 4; initGraph(Vc); 
addEdge(0,1,1); addEdge(1,2,2); addEdge(2,3,-6); addEdge(3,1,2); 
buildBFEdges(Vc); 
neg = bellmanFord(0, Vc, dBF); 
printf("Negative cycle: %s\n", neg ? "DETECTED" : "None"); 
return 0; 
}
/* Compile: gcc -O0 -o sssp sssp.c */