import pytest
import json
import logging
import re
import os
from rdflib import Graph, Namespace, RDF, URIRef, Literal
from rdflib.collection import Collection
from rdflib.exceptions import ParserError

from rdflib_canon import CanonicalizedGraph, PoisonedDatasetException
from tests import load_nquads

MF = Namespace("http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#")
RDFC = Namespace("https://w3c.github.io/rdf-canon/tests/vocab#")
RDFT = Namespace("http://www.w3.org/ns/rdftest#")

testdata = os.environ.get("RDF_CANON_TESTDATA", 'testdata')
g = Graph().parse(location=os.path.join(testdata, 'manifest.ttl'))

names: list = []
parameters: list = []

logging.basicConfig(level=logging.DEBUG)


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
    action = load_nquads(action_uri)
    too_many_calls = False
    try:
        output = CanonicalizedGraph(action)
    except PoisonedDatasetException:
        too_many_calls = True

    if type_ == RDFC.RDFC10NegativeEvalTest:
        assert too_many_calls
    elif type_ == RDFC.RDFC10EvalTest:
        assert not too_many_calls
        try:
            result = load_nquads(result_uri)
        except ParserError as e:
            pytest.xfail(e.msg)

        # Fail and print output
        assert set(output.canon) == set(result)

    elif type_ == RDFC.RDFC10MapTest:
        with open(str(result_uri.replace("file://", ""))) as f:
            result = json.load(f)
        assert result == output.issued
    else:
        raise AssertionError
