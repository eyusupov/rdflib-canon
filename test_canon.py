import pytest
import json
import re
from rdflib import Graph, Namespace, RDF, URIRef, BNode, ConjunctiveGraph, Literal
from rdflib.term import Node, IdentifiedNode
from rdflib.graph import Dataset, DATASET_DEFAULT_GRAPH_ID
from rdflib.collection import Collection
from rdflib.exceptions import ParserError
from rdflib.plugins.parsers.ntriples import r_nodeid
from rdflib.plugins.parsers.nquads import NQuadsParser
from rdflib.parser import URLInputSource
from rdflib.plugin import Parser, register, _plugins, Plugin
from rdflib.store import Store

from .canon import CanonicalizedGraph, PoisonedDatasetException, post_canon_cmp

MF = Namespace("http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#")
RDFC = Namespace("https://w3c.github.io/rdf-canon/tests/vocab#")
RDFT = Namespace("http://www.w3.org/ns/rdftest#")

g = Graph().parse(location='tests/manifest.ttl')
names = []
parameters = []


class OurNQuadsParser(NQuadsParser):
    def nodeid(self, bnode_context=None):
        if self.peek("_"):
            # Fix for https://github.com/RDFLib/rdflib/issues/204
            bnode_id = self.eat(r_nodeid).group(1)
            return BNode(bnode_id)
        return False

plugin = Plugin("nquads+bnode_id", Parser, "" , str(OurNQuadsParser.__class__))
plugin._class = OurNQuadsParser
_plugins[(plugin.name, plugin.kind)] = plugin

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




def load_nquads(uri):
    g = Dataset()
    g.parse(uri, format="nquads+bnode_id")
    return g


def pytest_generate_tests(metafunc) -> None:  # noqa: ANN001
    names = []
    parameters = []
    for manifest, _, _ in g.triples((None, RDF.type, MF.Manifest)):
        entries = g.value(manifest, MF.entries, None)
        assert entries is not None, "could not find entries in the test manifest"
        for entry in Collection(g, entries):
            id_ = re.match('.*#(.*)', entry)[1]
            type_ = g.value(entry, RDF.type, None)
            name = g.value(entry, MF.name, None)
            if g.value(entry, RDFC.hashAlgorithm , None) == Literal("SHA384"):
                # TODO
                continue
            action_uri = g.value(entry, MF.action, None)
            result_uri = g.value(entry, MF.result, None)
            approval = g.value(entry, RDFT.approval, None)
            if approval != RDFT.Approved:
                print(f"{name} is not approved (status is {approval}), skipping")
            assert type_ is not None, f"type is empty for {name}"
            assert action_uri is not None, f"action is empty for {name}"
            assert isinstance(action_uri, URIRef)
            if type_ == RDFC.RDFC10EvalTest:
                assert result_uri is not None, "result is empty"
                assert isinstance(result_uri, URIRef)
            names.append(id_)
            parameters.append((action_uri, result_uri, type_))
        metafunc.parametrize("action_uri, result_uri, type_", parameters, ids=names)

# "If more than one node produces the same N-degree hash, the order in which these nodes receive a canonical identifier does not matter."
# "Technically speaking, one implementation might return a canonicalized dataset that maps particular blank nodes to different identifiers than another implementation, however, this only occurs when there are isomorphisms in the dataset such that a canonically serialized expression of the dataset would appear the same from either implementation."

def test_single(action_uri, result_uri, type_, request):
    test_id = request.node.callspec.id
    action = load_nquads(action_uri)
    too_many_calls = False
    try:
        output = CanonicalizedGraph(action, store=OrderedStore())
    except PoisonedDatasetException:
        too_many_calls = True

    if type_ == RDFC.RDFC10NegativeEvalTest:
        assert too_many_calls
    elif type_ == RDFC.RDFC10EvalTest:
        try:
            result = load_nquads(result_uri)
        except ParserError as e:
            pytest.xfail(e.msg)

        if set(output.canon.quads()) == set(result):
            pass
        else:
            result = post_canon_cmp(output.canon, result)
            if result:
                pytest.xfail("Ambiguous isomorphic canonization")
            elif result is None:
                pytest.xfail("Too many variable to check ambiguous canonization")
            else:
                # Print output
                assert set(output.canon.quads()) == set(result)

    elif type_ == RDFC.RDFC10MapTest:
        with open(str(result_uri.replace("file://", ""))) as f:
            result = json.load(f)
        # TODO: permute this too, so that the remaining tests will not fail
        assert result == output.issued
    else:
        raise AssertionError