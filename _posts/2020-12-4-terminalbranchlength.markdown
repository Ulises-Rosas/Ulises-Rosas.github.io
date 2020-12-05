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

## Collapse nodes into a polytomy

Between two internal nodes theres is a edge (i.e., branch length) and when this edge is
so short, you might want to collapse these two nodes, thus creating a polytomy. 

Given mock the following tree:

```python
str_tree = '((A:1,B:2):3,((C:0,D:0):0,E:7):8);'
tree = dendropy.Tree.get_from_string(str_tree, 'newick')

print( tree.as_string('newick') )
```
```
((A:1.0,B:2.0):3.0,((C:0.0,D:0.0):0.0,E:7.0):8.0);
```

We can see from the above string that 'C' and 'D' form a single clade and 'E' is the sister taxon of these ones. Now, lets try to collapse these three taxa into a single clade by using a threshold (i.e., `min_edge = 0`):

```python
min_edge = 0

for nd in tree.postorder_edge_iter():
    if nd.length is None:
        continue
    if nd.is_internal() and nd.length == min_edge:
            nd.collapse()

print( tree.as_string('newick') )
```
```
((A:1.0,B:2.0):3.0,(C:0.0,D:0.0,E:7.0):8.0);
```
Notice that now 'C', 'D', and 'E' are inside the same parenthesis.


