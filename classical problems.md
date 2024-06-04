1.遍历树  build tree；sorting；recursion

```python
from collections import defaultdict
n = int(input())
tree = defaultdict(list)
parents = []
children = []
for i in range(n):
    t = list(map(int, input().split()))
    parents.append(t[0])
    if len(t) > 1:
        ch = t[1::]
        children.extend(ch)
        tree[t[0]].extend(ch)


def traversal(node):
    seq = sorted(tree[node] + [node])
    for x in seq:
        if x == node:
            print(node)
        else:
            traversal(x)


traversal((set(parents) - set(children)).pop()) #寻找根节点
```

2.pots

```python
from collections import deque
def bfs(A,B,C):
    que = deque()
    que.append(((0,0),[]))
    visited = set((0,0))
    while que:
        t = que.popleft()
        a, b = t[0][0], t[0][1]
        commands = t[1]
        if a == C or b == C:
            return commands
        visited.add((a,b))
        dir = [(0,b),(a,0),(A,b),(a,B)\
            ,(min(A,a+b),max(0,a+b-A)),(max(a+b-B,0),min(B,a+b))]
        for x in dir:
            if x not in visited:
                new_command = [translate(a,b,x)]
                que.append((x,commands+new_command))
    return ['impossible']
def translate(a, b, new):
    if new == (0,b):
        return 'DROP(1)'
    elif new == (a,0):
        return 'DROP(2)'
    elif new == (A,b):
        return 'FILL(1)'
    elif new == (a,B):
        return 'FILL(2)'
    elif new == (min(A,a+b),max(0,a+b-A)):
        return 'POUR(2,1)'
    elif new == (max(a+b-B,0),min(B,a+b)):
        return 'POUR(1,2)'
A, B, C = map(int, input().split())
output = bfs(A,B,C)
if output == ['impossible']:
    print(output[0])
else:
    print(len(output))
    for x in output:
        print(x)
```

3.冰阔落

```python
father = []
def find_father(x):
    if father[x] != x:
        father[x] = find_father(father[x])
    return father[x]
def check(x,y):
    if find_father(x) == find_father(y):
        return True
    else:
        return False
def join_father(x,y):
    father[find_father(y)] = father[x]



while True:
    try:
        n, m = map(int, input().split())
        father = [i for i in range(n+1)]
        for _ in range(m):
            a, b = map(int, input().split())
            if check(a,b):
                print('Yes')
            else:
                print('No')
                join_father(a, b)
        output = []
        for x in range(1,n+1):
            if find_father(x) == x:
                output.append(x)
        print(len(output))
        print(' '.join(map(str, output)))
    except EOFError:
        break
```

3.堆猪

```python
#懒删除
import heapq
from collections import defaultdict

out = defaultdict(int)
pigs_heap = []
pigs_stack = []

while True:
    try:
        s = input()
    except EOFError:
        break

    if s == "pop":
        if pigs_stack:
            out[pigs_stack.pop()] += 1
    elif s == "min":
        if pigs_stack:
            while True:
                x = heapq.heappop(pigs_heap)
                if not out[x]:
                    heapq.heappush(pigs_heap, x)
                    print(x)
                    break
                out[x] -= 1
    else:
        y = int(s.split()[1])
        pigs_stack.append(y)
        heapq.heappush(pigs_heap, y)
```

4.奶牛排队

```python
N = int(input())
heights = [int(input()) for _ in range(N)]
left_bound = [-1] * N
right_bound = [N] * N

stack = []  # 单调栈，存储索引

# 求左侧第一个≥h[i]的奶牛位置
for i in range(N):
    while stack and heights[stack[-1]] < heights[i]:
        stack.pop()
    if stack:
        left_bound[i] = stack[-1]
    stack.append(i)

stack = []  # 清空栈以供寻找右边界使用

# 求右侧第一个≤h[i]的奶牛位
for i in range(N-1, -1, -1):
    while stack and heights[stack[-1]] > heights[i]:
        stack.pop()
    if stack:
        right_bound[i] = stack[-1]
    stack.append(i)
ans = 0

for i in range(N):  # 枚举右端点 B寻找 A，更新 ans
    for j in range(left_bound[i] + 1, i):
        if right_bound[j] > i:
            ans = max(ans, i - j + 1)
            break
print(ans)
```