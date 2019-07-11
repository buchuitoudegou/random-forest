# Random Forest

## 基本思路
项目二的使用随机森林进行回归拟合，在建树的时候使用bagging的策略，提取一部分样本生成一棵回归树；在分裂节点的时候，也用随机的方法抽取一部分特征属性并选取其中最优属性进行划分。在预测时，综合每棵树的结果，进行加权平均，得到最终结果。

## 实验环境
- Ubuntu16.04
- Anaconda

## 实验步骤
### 节点划分
这是生成一棵回归树最重要的步骤。构造一棵树我们可以理解为递归节点划分的过程。
```python
left, right = node['group']
del(node['group'])
if not left or not right:
	node['left'] = node['right'] = to_terminal(left + right)
	return
if depth >= max_depth:
	node['left'], node['right'] = to_terminal(left), to_terminal(right)
	return
if len(left) <= min_size:
	node['left'] = to_terminal(left)
```
这里实际上是一个预剪枝的过程。通过判断叶子节点的样本数目是否大于最小样本数目，或者深度是否达到最大深度，来对当前节点进行剪枝：
1. 如果已经达到最大深度：作为叶子节点，不再进行划分
2. 如果当前节点的样本数少于最小样本数，不再进行划分。
这样做的目的是防止过拟合的发生，在样本数量大的时候，不设置树的深度会导致严重的过拟合。

```python
else:
    node['left'] = get_split(left, n_features)
    split(node['left'], max_depth, min_size, n_features, depth + 1)
if len(right) <= min_size:
    node['right'] = to_terminal(right)
else:
    node['right'] = get_split(right, n_features)
    split(node['right'], max_depth, min_size, n_features, depth + 1)
```
接下来实际上就是一个递归分割的过程，用get_split获得一个最佳分割属性。

### 获取最佳分割属性
```python
while len(cur_features) < log(2, n_features):
    index = randrange(len(dataset[0] - 1))
    if index not in cur_features:
        cur_features.append(index)
for index in cur_features:
    for row in dataset:
        groups = test_split(index, row[index], dataset)
        gini = gini_index(groups, class_values)
        if gini < s_score:
            s_index, s_value, s_score, s_groups = index, row[index], gini, groups
return { 'index': s_index, 'value': s_value, 'group': s_groups }

```
首先根据随机特征数获取要从里面得到需要筛选的属性。计算这些属性分割的gini系数，找到最小的gini系数的分割，即最佳分割。

这里的log是《机器学习》上推荐的随机样本属性数量。也可以设成参数的形式，自定义输入。

### 模型预测
```python
def predict(node, row):
    """
    description: make prediction with the tree, navigating it and get the output
    """
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']
```
使用递归的方法找到叶子节点并返回它的值，即这棵树的预测结果。最后将这些结果进行加权平均。

## 实验优化
### 利用python多进程的并行化
```python
from random import randint
from multiprocessing import Lock, Queue, Pool
lock = Lock()
result = Queue()
def append_tree(sample, max_depth, min_size, n_features):
    tree = build_tree(sample, max_depth, min_size, n_features)
    result.put(tree)   
p = Pool()
p.close()
p.join()


def multi_process_random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    trees = []
    for i in range(n_trees):
        sample = sub_sample(train, sample_size)
        p.apply_async(build_tree, args=(sample, max_depth, min_size, n_features,))
    p.close()
    p.join()
```
思路比较简单，利用多进程模块，在每一棵树训练的时候增加一个子进程。如果考虑到计算机性能问题，可以利用进程池的方式，规定最大进程数。

## 实验结果
控制变量：
- 最大深度：10
- 叶子节点最少样本数：1
### Project2前10000个训练数据（单进程）
|模型|决策树数目|训练时间|score|
|:----:|:----:|:----:|:----:|
|myRF|1|1min2s|-0.0231|
|sklearn|1|<1s|0.03348|

### Project2前10000个训练数据（多进程）
|模型|决策树数目|训练时间|score|
|:----:|:----:|:----:|:----:|
|myRF|10|7min21s|-0.0014|
|sklearn|10|11s|0.05327|

## 实验总结
相较于sklearn的随机森林，自己写的随机森林显然有很大的不足。具体有以下几个方面可以进行优化：
1. 参数数量。sklearn随机森林的参数数量很多，例如最大特征数、随机数等。
2. 训练速度优化。即使用了多进程的训练方式，自己实现的随机森林的训练速度依然慢得惊人，对于Project2这种1000w+数据集的项目而言，没有用武之地。
3. 训练结果优化。对于自己写的随机森林，过拟合的现象依然存在，从而导致模型在测试集合上的表现不佳。这一点需要在算法和参数上进行调整优化。
4. 利用cython的优化。关闭python的模块检查、类型检查等，提高代码的执行速度。