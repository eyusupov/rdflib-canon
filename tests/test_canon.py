import pytest
import json
import re
import os
from rdflib import Graph, Namespace, RDF, URIRef, BNode, Literal
from rdflib.graph import Dataset
from rdflib.collection import Collection
from rdflib.exceptions import ParserError
from rdflib.plugins.parsers.ntriples import r_nodeid
from rdflib.plugins.parsers.nquads import NQuadsParser
from rdflib.plugin import Parser, _plugins, Plugin

from itertools import permutations
from rdflib_canon import CanonicalizedGraph, PoisonedDatasetException, permute_ds_bnodes, permute_ds_bnode_ids, TooManyPermutations

MF = Namespace("http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#")
RDFC = Namespace("https://w3c.github.io/rdf-canon/tests/vocab#")
RDFT = Namespace("http://www.w3.org/ns/rdftest#")

testdata = os.environ.get("RDF_CANON_TESTDATA", 'testdata')
g = Graph().parse(location=os.path.join(testdata, 'manifest.ttl'))

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

def test_single(action_uri, result_uri, type_, request):
    test_id = request.node.callspec.id
    action = load_nquads(action_uri)
    too_many_calls = False
    try:
        output = CanonicalizedGraph(action)
    except PoisonedDatasetException:
        too_many_calls = True

    if type_ == RDFC.RDFC10NegativeEvalTest:
        assert too_many_calls
    elif type_ == RDFC.RDFC10EvalTest:
        try:
            result = load_nquads(result_uri)
        except ParserError as e:
            pytest.xfail(e.msg)

        if set(output.canon.quads()) != set(result):
            try:
                # See the last note in the section 4.4.3 of the spec.
                for ds in permute_ds_bnodes(output.canon):
                    if set(ds) == set(result):
                        pytest.xfail("Ambiguous isomorphic canonization")
            except TooManyPermutations:
                pytest.xfail("Too many variable to check ambiguous canonization")
        # Fail and print output
        assert set(output.canon.quads()) == set(result)

    elif type_ == RDFC.RDFC10MapTest:
        with open(str(result_uri.replace("file://", ""))) as f:
            result = json.load(f)
        if result != output.issued:
            if len(output.issued.keys()) < 7:
                for perm in permutations(output.issued.keys()):
                    if result == dict(zip(perm, output.issued.values())):
                        pytest.xfail("Ambiguous isomorphic canonization")
        assert result == output.issued
    else:
        raise AssertionError
