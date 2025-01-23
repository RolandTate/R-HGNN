import scipy.io
import urllib.request
import os
from dgl.data.utils import download, get_download_dir, _get_dgl_url
import dgl

url = 'dataset/ACM.mat'
data_path = get_download_dir() + '/ACM.mat'
if not os.path.exists(data_path):
    download(_get_dgl_url(url), path=data_path)
data = scipy.io.loadmat(data_path)
print(list(data.keys()))

print(type(data['PvsA']))
print('#Papers:', data['PvsA'].shape[0])
print('#Authors:', data['PvsA'].shape[1])
print('#Links:', data['PvsA'].nnz)

pa_g = dgl.from_scipy({('paper', 'written-by', 'author'): data['PvsA']})
# equivalent (shorter) API for creating heterograph with two node types:
# pa_g = dgl.bipartite(data['PvsA'], 'paper', 'written-by', 'author')

print('Node types:', pa_g.ntypes)
print('Edge types:', pa_g.etypes)
print('Canonical edge types:', pa_g.canonical_etypes)

# 节点和边都是从零开始的整数ID，每种类型都有其自己的计数。要区分不同类型的节点和边缘，需要指定类型名称作为参数。
print(pa_g.number_of_nodes('paper'))

# 如果规范边类型名称是唯一可区分的，则可以将其简化为边类型名称。
print(pa_g.number_of_edges(('paper', 'written-by', 'author')))
print(pa_g.number_of_edges('written-by'))
## 获得论文#1 的作者
print(pa_g.successors(1, etype='written-by'))