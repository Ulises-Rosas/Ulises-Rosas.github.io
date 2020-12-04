---
layout: posts
title: Branches and dendropy
date:   2020-12-4
categories: jekyll update
excerpt: "Code gist"
---

Dendropy contains many subclasses inside each object, then extract information might actually
end up being a task of finding the correct one subclass. Here I present the call of those subclasses
for getting some common metrics in branches

## Terminal branch lengths

Given the following mock tree:

```python
import dendropy

str_tree = '((A:1,B:2):3,(C:4,D:5):6);'
tree = dendropy.Tree.get_from_string(str_tree, 'newick')

print( tree.as_ascii_plot(plot_metric = 'length') )
```

```
        /-- A
/------+
|       \---- B
+              
|             /-------- C 
\------------+           
              \---------- D
```
We can get terminal branch length by iterating nodes in 'postorder':

```python
df = {}
for nd in tree.postorder_edge_iter():
    if nd.length is None:
        continue
    taxn = nd.head_node.taxon
    if taxn:
        df[taxn.label] = nd.length

df
```

```
{'A': 1.0, 'B': 2.0, 'C': 4.0, 'D': 5.0}
```