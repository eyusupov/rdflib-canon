from rdflib import Graph, BNode
from rdflib.graph import Dataset
from rdflib.term import Node, IdentifiedNode
from rdflib.plugins.parsers.ntriples import r_nodeid
from rdflib.plugins.parsers.nquads import NQuadsParser
from rdflib.plugin import Parser, _plugins, Plugin
from rdflib.store import Store


class OurNQuadsParser(NQuadsParser):
    def nodeid(self, bnode_context=None):
        if self.peek("_"):
            # Fix for https://github.com/RDFLib/rdflib/issues/204
            bnode_id = self.eat(r_nodeid).group(1)
            return BNode(bnode_id)
        return False

class OrderedStore(Store):
    context_aware = True
    graph_aware = True

    def __init__(self, configuration=None, identifier=None):
        super().__init__(configuration, identifier)
        self.store = []

    def add(
        self,
        triple: tuple[Node, Node, Node],
        context: Graph | None,
        quoted: bool = False,
    ):
        s, p, o = triple
        quad = (s, p, o, context)
        if quad in self.store:
            return
        self.store.append(quad)

    def add_graph(self, g):
        for triple in g:
            self.add(triple, g)

    def remove(self, triple, context=None):
        raise NotImplementedError

    def triples(
        self,
        triple_pattern: tuple[
            IdentifiedNode | None, IdentifiedNode | None, Node | None
        ],
        context=None,
    ):
        if triple_pattern != (None, None, None):
            raise NotImplementedError
        for s, p, o, c in self.store:
            if context is not None and c != context:
                continue
            yield (s, p, o), [c]

    def __len__(self, context=None):
        return len(filter(lambda s, p, o, c: c == context, self.store))

    def contexts(self, triple=None):
        if triple:
            raise NotImplementedError
        for c in sorted(set(map(lambda s, p, o, c: c, self.store))):
            yield c


def load_nquads(*args, **kwargs):
    g = Dataset()
    if format not in kwargs:
        kwargs["format"] = "nquads+bnode_id"
    g.parse(*args, **kwargs)
    return g

plugin = Plugin("nquads+bnode_id", Parser, "" , str(OurNQuadsParser.__class__))
plugin._class = OurNQuadsParser
_plugins[(plugin.name, plugin.kind)] = plugin

