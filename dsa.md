1.KMP

```python
def prefix_function(s):
    n = len(s)
    pi = [0] * n  # 初始化前缀函数数组
    for i in range(1, n): 
        j = pi[i - 1]  # 获取前一个位置的前缀函数值
        while j > 0 and s[i] != s[j]:
            j = pi[j - 1]
        if s[i] == s[j]:
            j += 1
        pi[i] = j
    return pi

#实现1:构造一个数组，计算前缀函数值等于s长度的个数
def find_occurrences(s, t):
    cur = s + "*" + t 
    ans = []
    lps = prefix_function(cur)
    for i in range(len(s) + 1, len(cur)):
        if lps[i] == len(s):
            ans.append(i - 2 * len(s))
    return ans #输出含有目标字符串的起始位置的列表
 
#实现2：当出现匹配失败时，可以跳过长度为前缀函数的部分继续重新匹配
def find_occurrences(s, t):
    lps = prefix_function(s)
    ans = []
    j = 0 
    for i in range(len(t)):
        while j > 0 and t[i] != s[j]:
            j = lps[j-1]
        if t[i] == s[j]:
            j += 1
        if j == len(s):
            ans.append(i - j + 1)
            j = lps[j-1]
     return ans
  
```

2.shunting yard（中序表达式转后序）

```python
precedence = {'+':1, '-':1, '*':2, '/':2}
for _ in range(int(input())):
    s = input().strip()
    num = ''
    op = []
    operator = []
    for i in s:
        if not i in precedence.keys() and not i in ['(',')']:
            num = num + i
        else:
            if num:
                if '.' in num:
                    op.append(float(num))
                else:
                    op.append(int(num))
            num = ''
            if i == '(':
                operator.append(i)
            elif i == ')':
                while operator[-1] != '(':
                    op.append(operator.pop(-1))
                operator.pop()
            else:
                while operator and operator[-1] != '(' and precedence[i] <= precedence[operator[-1]]:
                    op.append(operator.pop())
                operator.append(i)
    if len(num) != 0:
        if '.' in num:
            op.append(float(num))
        else:
            op.append(int(num))
    if operator:
        while operator:
            op.append(operator.pop())
    print(' '.join(map(str,op)))
```

3.disjointset

```python
class UnionFind:
    def __init__(self,n):
        self.p=list(range(n))
        self.h=[0]*n
    def find(self,x):
        if self.p[x]!=x:
            self.p[x]=self.find(self.p[x])
        return self.p[x]
    def union(self,x,y):
        rootx=self.find(x)
        rooty=self.find(y)
        if rootx!=rooty:
            if self.h[rootx]<self.h[rooty]:
                self.p[rootx]=rooty
            elif self.h[rootx]>self.h[rooty]:
                self.p[rooty]=rootx
            else:
                self.p[rooty]=rootx
                self.h[rootx]+=1
```

4.kruskal&prim

```python
#kruskal加边
uf=UnionFind(n)
edges.sort()
ans=0
for w,u,v in edges:
    if uf.union(u,v):
        ans+=w
print(ans)

#prim加点，同Dijkstra
visited=[0]*n
q=[(0,0)]
ans=0
while q:
    w,u=heappop(q)
    if visited[u]:
        continue
    ans+=w
    visited[u]=1
    for v in range(n):
        if not visited[v] and graph[u][v]!=-1:
            heappush(q,(graph[u][v],v))
print(ans)
```

5.Dijkstra

```python
def dijkstra(start,end):
    heap=[(0,start,[start])]
    visited = set()
    while heap:
        (cost,u,path)=heappop(heap)
        if u in visited: continue
        vis.add(u)
        if u==end: return (cost,path)
        for v in graph[u]:
            if v not in visited:
                heappush(heap,(cost+graph[u][v],v,path+[v]))
```

6.拓扑排序

```python
from collections import deque
def topo_sort(graph):
    in_degree={u:0 for u in graph}
    for u in graph:
        for v in graph[u]:
            in_degree[v]+=1
    q=deque([u for u in in_degree if in_degree[u]==0])
    topo_order=[]
    while q:
        u=q.popleft()
        topo_order.append(u)
        for v in graph[u]:
            in_degree[v]-=1
            if in_degree[v]==0:
                q.append(v)
    if len(topo_order)!=len(graph):
        return []  
    return topo_order
```

7.kosaraju

```python
def dfs1(u):
    vis[u] = True
    for v in g[u]:
        if vis[v] == False:
            dfs1(v)
    s.append(u）
def dfs2(u):
    color[u] = sccCnt #涂色
    for v in g2[u]:
        if color[v] == False:
            dfs2(v)
def kosaraju(u):
    sccCnt = 0
    for i in range(1, n + 1):
        if vis[i] == False:
            dfs1(i)
    for i in range(n, 0, -1):
        if color[s[i]] == False:
            sccCnt = sccCnt + 1
            dfs2(s[i])
```

