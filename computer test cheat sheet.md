# basic code summary (computer test cheat sheet)
## frequent use basic code
```python
itertools collections heapq math numpy
if..elif..else for...in while 
pop() sort() sorted() map(func, *iterables) abs() max() min() sum() math.pow(x, y) math.floor() math.ceil() len() append() extend() insert() sort() list() str.replace() str.split() str.strip() str.join() set() dict() range() ord() chr() heapq.heappush(heap,item) heapq.heappop(heap) heapq.heappushpop(heap,item) set([]) math.log(x,[base=math.e]) math.factorial

matrix = [[0 for _ in range(n)] for _ in range(m)] maze.append([-1] + [int(_) for _ in input().split()] + [-1])
sorted(list, lambda x: x[0])
# pylint: skip-file
f"xxxx{value:.2f}" "{:.2f}".format(num)
print(' '.join(map(str, minStep[i])))
#通过值来搜索键  
#找到所有的键:  
#法一:
keys = [key for key, value in my_dict.items() if value == search_value] 
#法二:
keys = list(filter(lambda key: my_dict[key] == search_value, my_dict)) 
#找到第一个符合条件的键:  
key = next((key for key, value in my_dict.items() if value == search_value), None)
#向字典中某一个键下添加元素:  
my_dict = {'key1': [1, 2, 3], 'key2': [4, 5]} 
my_dict['key1'].append(4)
#删除键值对  
#法一:
del my_dict['city'] 
#法二:
age = my_dict.pop('age')
#遍历字典:  
# 遍历键
or key in my_dict:
# 遍历值  
for value in my_dict.values():
	print(value)
# 遍历键值对  
for key, value in my_dict.items():
	print(f"{key}: {value}")
#字典推导式举例:  
numbers = [1, 2, 3, 4, 5]  
squared_dict = {n: n**2 for n in numbers} 
print(squared_dict)
sorted_dict = dict(sorted(my_dict.items(), key=lambda x: x[1], reverse=True))
b = str(format(n,'b'))
b[::-1] #reverse
itertools.permutations(arr, 2(c这里是长度))
a = b[:]
import calendar, datetime print(calendar.isleap(2020)) # 输出: True
print(datetime.datetime(2023, 10, 5).weekday()) # 输出: 3 (星期四)
squared = list(map(lambda x: x**2, [1, 2, 3, 4]))
print("\n".join(list)) print(*line)
number % 1 == 0 #whole number? 
res = filter(lambda x: x > 0, a) #filter out positive
numpy.transpose() numpy.array() numpy.T()
for index, char in enumerate(iterable) #return unique index with element (can get element without repeat not only the first would found)
```
## algorithm and data structures
### binary search
```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
			return mid 
		elif arr[mid] < target:
            left = mid + 1
        else:
			right = mid - 1
		return -1
```
### two pointer
```python
l=0 
r=n-1
while l < r:
    if a[l] + a[r] == M:
        print(a[l], a[r])
        l += 1
        r -= 1
    elif a[l] + a[r] < M:
        l += 1
else:
r -= 1
```
### recursion
```python
importsys
sys.setrecursionlimit(1 << 30)
from functools import lru_cache
@lru_cache(maxsize=None)
def recursive(n):
    if n == 0:
        return 1
    elif n == 1:
        return 1
    else:
        return recursive(n - 1) + recursive(n - 2)
```
### dfs
```python
def dfs(x, y):
	field[x][y] = '.'
	for dx, dy in directions:
		nx, ny = x + dx, y + dy
			if 0 <= nx < n and 0 <= ny < m and field[nx][ny] == 'W':
				dfs(nx, ny)

n, m = map(int, input().split())
field = [list(input()) for _ in range(n)]
directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
cnt = 0

for i in range(n):
    for j in range(m):
        if field[i][j] == 'W':
print(cnt)
```
迷宫的可行路径数(从左上角到右下角)
```python
MAXN = 5
n, m = map(int, input().split())
maze = []
for _ in range(n):
    row = list(map(int, input().split()))
    maze.append(row)
visited = [[False for _ in range(m)] for _ in range(n)]
counter = 0
MAXD = 4
dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]
def is_valid(x, y):
    return 0 <= x < n and 0 <= y < m and maze[x][y] == 0 and not visited[x][y]
def DFS(x, y):
    global counter
	if x == n - 1 and y == m - 1:#这里可以改成终点 
		counter += 1
        return
    visited[x][y] = True
    for i in range(MAXD):
        nextX = x + dx[i]
        nextY = y + dy[i]
        if is_valid(nextX, nextY):
            DFS(nextX, nextY)
    visited[x][y] = False
DFS(0, 0)#这里可以改为起点，实现任意化 
print(counter)
```
指定步数迷宫(问能否从左上角到右下角)
```python
MAXN = 5
n, m, k = map(int, input().split())
maze = []
for _ in range(n):
    row = list(map(int, input().split()))
    maze.append(row)
visited = [[False for _ in range(m)] for _ in range(n)]
canReach = False
MAXD = 4
dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]
def is_valid(x, y):
    return 0 <= x < n and 0 <= y < m and maze[x][y] == 0 and not visited[x][y]
def DFS(x, y, step):#将step放在dfs中实现递归调用 
	global canReach
    if canReach:
        return
	if x == n - 1 and y == m - 1:#可改为终点 
		if step == k:
            canReach = True
        return
    visited[x][y] = True
    for i in range(MAXD):
        nextX = x + dx[i]
        nextY = y + dy[i]
        if step < k and is_valid(nextX, nextY):
            DFS(nextX, nextY, step + 1)
    visited[x][y] = False
DFS(0, 0, 0)#可改为起点
print("Yes" if canReach else "No")
```
矩阵最大权值路径(注意回溯和列表的拷贝)
```python
n, m = map(int, input().split())
maze = [list(map(int, input().split())) for _ in range(n)]
directions = [(0, 1), (1, 0), (0, -1), (-1, 0)] # 右、下、左、上 
visited = [[False] * m for _ in range(n)] # 标记访问
max_path = []
max_sum = -float('inf') # 最大权值初始化为负无穷
def dfs(x, y, current_path, current_sum):
    global max_path, max_sum
	# 到达终点，更新结果
	if (x, y) == (n - 1, m - 1):
        if current_sum > max_sum:
            max_sum = current_sum
            max_path = current_path[:]
		return #return 后面没有值，表示函数直接结束并返回 None，退出递归调用，回溯的关键。
	for dx, dy in directions:
        nx, ny = x + dx, y + dy
		# 检查边界和是否访问过
		if 0 <= nx < n and 0 <= ny < m and not visited[nx][ny]:
		visited[nx][ny] = True current_path.append((nx, ny))
		# 递归搜索
		dfs(nx, ny, current_path, current_sum + maze[nx][ny])
		# 回溯 
		current_path.pop() 
		visited[nx][ny] = False
# 初始化起点
visited[0][0] = True
dfs(0, 0, [(0, 0)], maze[0][0])
for x, y in max_path:
    print(x + 1, y + 1)
```
dfs与dp结合。例题滑雪——反向找到递增
```python
r, c = map(int, input().split())
matrix = [list(map(int, input().split())) for _ in range(r)]
dp = [[0 for _ in range(c)] for _ in range(r)]
def dfs(x, y):
    dx = [0, 0, 1, -1]
    dy = [1, -1, 0, 0]
    for i in range(4):
        nx, ny = x + dx[i], y + dy[i]
        if 0 <= nx < r and 0 <= ny < c and matrix[x][y] > matrix[nx][ny]:
            if dp[nx][ny] == 0:
                dfs(nx, ny)
            dp[x][y] = max(dp[x][y], dp[nx][ny] + 1)
    if dp[x][y] == 0:
        dp[x][y] = 1
max_len=0
for i in range(r):
    for j in range(c):
        if not dp[i][j]:
            dfs(i, j)
        max_len = max(max_len, dp[i][j])
print(max_len)
```
### bfs
```python
from collections import deque
def bfs(s, e):
    inq = set()
    inq.add(s)
    q = deque()
    q.append((0, s))
		while q:
				now, top = q.popleft() 
				if top == e:
						return now
```
迷宫最短路径问题
```python
from collections import deque
MAX_DIRECTIONS = 4
dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]
def is_valid_move(x, y, n, m, maze, in_queue):
    return 0 <= x < n and 0 <= y < m and maze[x][y] == 0 and (x, y) not in in_queue
def bfs(start_x, start_y, n, m, maze):
    queue = deque()
    queue.append((start_x, start_y))
    in_queue = set()
    prev = [[(-1, -1)] * m for _ in range(n)]
    in_queue.add((start_x, start_y))
    while queue:
        x, y = queue.popleft()
        if x == n - 1 and y == m - 1:
            return prev
        for i in range(MAX_DIRECTIONS):
            next_x = x + dx[i]
            next_y = y + dy[i]
            if is_valid_move(next_x, next_y, n, m, maze, in_queue):
                prev[next_x][next_y] = (x, y)
                in_queue.add((next_x, next_y))
                queue.append((next_x, next_y))
	return None
def print_path(prev, end_pos):
    path = []
    while end_pos != (-1, -1):
        path.append(end_pos)
        end_pos = prev[end_pos[0]][end_pos[1]]
    path.reverse()
	for pos in path:
        print(pos[0] + 1, pos[1] + 1)
if __name__ == '__main__':
    n, m = map(int, input().split())
    maze = [list(map(int, input().split())) for _ in range(n)]
    prev = bfs(0, 0, n, m, maze)
    if prev:
        print_path(prev, (n - 1, m - 1))
    else:
        print("No path found")
```
跨步迷宫
```python
from collections import deque
MAXD = 8
dx = [0, 0, 0, 0, 1, -1, 2, -2]
dy = [1, -1, 2, -2, 0, 0, 0, 0]
def canVisit(x, y, n, m, maze, in_queue):
    return 0 <= x < n and 0 <= y < m and maze[x][y] == 0 and (x, y) not in in_queue
def bfs(start_x, start_y, n, m, maze):
    q = deque()
    q.append((0, start_x, start_y))  # (step, x, y)
    in_queue = {(start_x, start_y)}
    while q:
        step, x, y = q.popleft()
        if x == n - 1 and y == m - 1:
            return step
        for i in range(MAXD):
            next_x = x + dx[i]
            next_y = y + dy[i]
            next_half_x = x + dx[i] // 2
            next_half_y = y + dy[i] // 2
            if canVisit(next_x, next_y, n, m, maze, in_queue) and maze[next_half_x][next_half_y] == 0:
                in_queue.add((next_x, next_y))
                q.append((step + 1, next_x, next_y))
	return -1
if __name__ == '__main__':
    n, m = map(int, input().split())
    maze = [list(map(int, input().split())) for _ in range(n)]
    step = bfs(0, 0, n, m, maze)
    print(step)
```
矩阵中的块
```python
from collections import deque
MAXD = 4
dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]
def bfs(x, y):
    q = deque([(x, y)])
    inq_set.add((x,y))
    while q:
        front = q.popleft()
        for i in range(MAXD):
            next_x = front[0] + dx[i]
            next_y = front[1] + dy[i]
            if matrix[next_x][next_y] == 1 and (next_x,next_y) not in inq_set:
                inq_set.add((next_x, next_y))
                q.append((next_x, next_y))
n, m = map(int, input().split())
matrix=[[-1]*(m+2)]+[[-1]+list(map(int,input().split()))+[-1] for i in range(n)]+[[-1]*
(m+2)]
inq_set = set()
counter = 0
for i in range(1,n+1):
    for j in range(1,m+1):
        if matrix[i][j] == 1 and (i,j) not in inq_set:
			bfs(i, j)
            counter += 1
print(counter)
```
### dp
双dp(土豪购物)——两种情况分析，取或不取，放或不放
```python
def max_value(s):
    n=len(s)
    dp1=[0 for _ in range(n)]
    dp2=[0 for _ in range(n)]
    dp1[0]=s[0]
    dp2[0]=s[0]
	for i in range(1,n):
		dp1[i]=max(dp1[i-1]+s[i],s[i])#不放回 
		dp2[i]=max(dp2[i-1]+s[i],dp1[i-1],s[i])#放回之前某个，放回现在也就是第i个，从现在开始取
    return max(max(dp1),max(dp2))
s=list(map(int,input().split(',')))
max_num=max_value(s)
print(max_num)
```
背包问题(小偷背包——01背包取或不取)
```python
n,b=map(int, input().split())
price=[int(i) for i in input().split()] #(注意这里下标从零开始)
weight=[int(i) for i in input().split()]
bag=[[0]*(b+1) for _ in range(n+1)]
for i in range(1,n+1):
    for j in range(1,b+1):
        if weight[i]<=j:
            bag[i][j]=max(price[i]+bag[i-1][j-weight[i]], bag[i-1][j])
        else:
            bag[i][j]=bag[i-1][j]
print(bag[-1][-1])
```
完全背包(无数个)
```python
n, a, b, c = map(int, input().split())
dp = [float('-inf')]*n
for i in range(1, n+1):
    for j in (a, b, c):
        if i >= j:
	        dp[i] = max(dp[i-j] + 1, dp[i])
print(dp[n])
```
## others
```python
#spiraling
n = int(input())
matrix = [[0]*n for i in range(n)]
l,r,u,d = 0,n-1,0,n-1
number = 1
while l <= r and u <= d:
    for i in range(l, r + 1):
        matrix[u][i] = number
        number += 1
    u += 1
    for i in range(u, d + 1):
        matrix[i][r] = number
        number += 1
    r -= 1
    if u <= d:
        for i in range(r, l - 1, -1):
            matrix[d][i] = number
            number += 1
        d -= 1
    if l <= r:
        for i in range(d, u - 1, -1):
            matrix[i][l] = number
            number += 1
        l += 1
for i in range(n):
    print(*matrix[i])

#onion
n = int(input())
matrix = [list(map(int,input().split())) for i in range(n)]
output = 0
u,d,l,r = 0,n-1,0,n-1
while l < r and u < d:
    sum_layer = 0
    for j in range(u,d+1):
        sum_layer += matrix[u][j] + matrix[d][j]
    u += 1
    d -= 1
    for j in range(l+1,r):
        sum_layer += matrix[j][l] + matrix[j][r]
    l += 1
    r -= 1
    output = max(output,sum_layer)
if n % 2 != 0:
    output = max(output,matrix[n//2][n//2])
print(output)

# 朴素法(时间复杂度是O(N^2)) (prime)
primesNumber = []
def is_prime(n):
    for i in range(2, n-1):
        if n % i == 0:
            return False
    return True
def primes(number):
    for i in range(2, number):
        if is_prime(i):
            primesNumber.append(i)
primes(10000)
print(primesNumber)

# 埃氏筛 (prime)
def sieve_of_eratosthenes(n):
# 创建一个布尔列表，初始化为 True，表示所有数字都假设为素数 
		primes = [True] * (n + 1)
		primes[0] = primes[1] = False # 0 和 1 不是素数
		# 从 2 开始，处理每个数字
		for i in range(2, int(n**0.5) + 1):
				if primes[i]: # 如果 i 是素数
						# 将 i 的所有倍数标记为非素数
						for j in range(i * i, n + 1, i):
                primes[j] = False
		# 返回所有素数
		return [x for x in range(2, n + 1) if primes[x]]

#欧拉筛 (prime)
# 返回小于r的素数列表 
def oula(r):
		# 全部初始化为0
		prime = [0 for i in range(r+1)] 
		# 存放素数
		common = []
		for i in range(2, r+1):
        if prime[i] == 0:
            common.append(i)
        for j in common:
            if i*j > r:
                break
            prime[i*j] = 1
						#将重复筛选剔除 
						if i % j == 0:
                break
    return common
prime = oula(20000)
print(prime)

#bubble sort
for i in range(n):
    for j in range(n-1-i):
        if l[j] + l[j+1] > l[j+1] + l[j]:
            l[j],l[j+1] = l[j+1],l[j]

#merge
def merge(A, B):
    i, j = 0, 0
	c = []
	while i < len(A) and j < len(B):
        if A[i] <= B[j]:
            c.append(A[i])
			i += 1
		else:
			c.append(B[j])
            j += 1
	# 将 A 的剩余元素加入 c
	c.extend(A[i:])  
	# 将 B 的剩余元素加入 c
	c.extend(B[j:])
    return len(c), c

#quick sort
def QuickSort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[0]  # Choose the first element as the pivot
        left = [x for x in arr[1:] if x < pivot]
        right = [x for x in arr[1:] if x >= pivot]
        return QuickSort(left) + [pivot] + QuickSort(right)

#monotonic stack
def next_greater_element(nums):
    stack = []
    result = [0] * len(nums)
	for i in range(len(nums)):  
		# 当栈不为空且当前考察的元素大于栈顶元素时  
		while stack and nums[i] > nums[stack[-1]]:
            index = stack.pop()
			result[index] = nums[i] 
		# 将当前元素的索引压入栈中 
		stack.append(i)
	# 对于栈中剩余的元素，它们没有更大的元素 
	while stack:
        index = stack.pop()
        result[index] = -1
    return result

#power of two
def isPowerOfTwo(n):
    if (n == 0):
        return No
    while (n != 1):
        if (n % 2 != 0):
            return No
        n = n // 2
    return Yes

#happy number
def numSquareSum(n):
    squareSum = 0
    while (n != 0):
        squareSum += (n % 10) * (n % 10)
        n = n // 10
    return squareSum
def isHappyNumber(n):
    st = set()
    while (1):
        n = numSquareSum(n)
        if (n == 1):
            return True
        if n in st:
            return False
        st.add(n)
print(isHappyNumber(n))

#perfect square
if math.sqrt(n) == math.floor(math.sqrt(i))

#strong numbers
import math
def strong_number(list):
    new_list = []
    for x in list:
        temp = x
        sum = 0
        while(temp):
            rem = temp % 10
            sum += math.factorial(rem)
            temp = temp // 10
        if(sum == x):
            new_list.append(x)
        else:
	        pass
    return new_list

#perfect number
def IsPerfectNumberFilterMethod(number):
    divisors = list(filter(lambda x: number % x == 0, range(1, number)))
    return sum(divisors) == number
if IsPerfectNumberFilterMethod(numer):
    print("yes")
else:
    print("no")
```
