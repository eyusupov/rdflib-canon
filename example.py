# coding: utf-8
import logging
from rdflib_canon import CanonicalizedGraph
from tests import load_nquads

logging.basicConfig(level=logging.DEBUG)
ds = load_nquads(data="""
<http://example.com/#p> <http://example.com/#q> _:e0 .
<http://example.com/#p> <http://example.com/#q> _:e1 .
_:e0 <http://example.com/#p> _:e2 .
_:e1 <http://example.com/#p> _:e3 .
_:e2 <http://example.com/#r> _:e3 .
""", format="nquads+bnode_id")
cds = CanonicalizedGraph(ds)
