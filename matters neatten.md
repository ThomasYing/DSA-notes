1.深copy和浅copy的问题

2.defaultdict方法

```python
from collections import defaultdict
my_list = [1, 2, 3, 2, 4, 1, 5, 6, 7, 6, 8]
count_dict = defaultdict(int)
# 遍历列表,统计每个元素的个数（默认值为0）
for item in my_list:
    count_dict[item] += 1
```

3.常用方法

Itertools:全排列permuations、笛卡尔积product、累次运算reduce

enumerate枚举列表，提供元素及其索引