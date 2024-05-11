# coding: utf-8
import logging
from rdflib_canon import CanonicalizedGraph
from tests import load_nquads

logging.basicConfig(level=logging.DEBUG)
ds = load_nquads("file:///home/eyusupov/sem/sources/rdf-canon/tests/rdfc10/test044-in.nq")
cds = CanonicalizedGraph(ds)
print(cds.calls)
