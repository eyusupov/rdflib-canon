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


def load_nquads(*args, **kwargs):
    g = Dataset()
    if format not in kwargs:
        kwargs["format"] = "nquads+bnode_id"
    g.parse(*args, **kwargs)
    return g


plugin = Plugin("nquads+bnode_id", Parser, "" , str(OurNQuadsParser.__class__))
plugin._class = OurNQuadsParser
_plugins[(plugin.name, plugin.kind)] = plugin

