
def spans_to_tree(spans, tokens):
    length = len(tokens)

    # Add missing spans.
    span_set = set(spans)
    for pos in range(length):
        if pos not in span_set:
            spans.append((pos, 1))

    spans = sorted(spans, key=lambda x: (x[1], x[0])) # pos, level
    pos_to_node = {}
    # root_node = None

    for i, span in enumerate(spans):

        pos, size = span

        if i < length:
            assert i == pos
            node = (pos, size, tokens[i])
            pos_to_node[pos] = node
            continue

        node = (pos, size, [])

        for i_pos in range(pos, pos+size):
            child = pos_to_node[i_pos]
            c_pos, c_size = child[0], child[1]

            if i_pos == c_pos:
                node[2].append(child)
            pos_to_node[i_pos] = node

    def helper(node):
        pos, size, tok = node
        if isinstance(tok, int):
            return tok
        return tuple([helper(x) for x in tok])

    root_node = pos_to_node[0]
    tree = helper(root_node)

    return tree


class TreesFromDiora(object):
    def __init__(self, net):
        self.diora = net

    def to_spans(self, lst):
        return [(pos, level + 1) for level, pos in lst]

    def parse_batch(self, batch_map):
        batch_size, length = batch_map['sentences'].shape
        root_level = length - 1
        tokens = [i for i in range(length)]

        trees = []
        for i_b in range(batch_size):
            spans = self.to_spans(self.diora.inside_tree[(i_b, 0)][(root_level, 0)])
            binary_tree = spans_to_tree(spans, tokens)
            trees.append(binary_tree)
        return trees
