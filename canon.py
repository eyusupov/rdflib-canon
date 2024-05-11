from collections import OrderedDict
from collections.abc import Iterable
from rdflib import Graph, BNode, Literal
from rdflib.term import IdentifiedNode
from rdflib.graph import Dataset, ConjunctiveGraph, _QuadType, DATASET_DEFAULT_GRAPH_ID
from rdflib.plugins.serializers.nt import _quoteLiteral
from hashlib import sha256
from itertools import permutations


def canon_ser_node(id_, n):
    if isinstance(n, Literal):
        return _quoteLiteral(n)
    if isinstance(n, BNode):
        if str(n) == id_:
            return "_:a"
        else:
            return "_:z"
    else:
        return n.n3()


def canon_nq_row(id_, quad):
    c = quad[3]
    if isinstance(c, Graph):
        c = c.identifier
    if c == DATASET_DEFAULT_GRAPH_ID or c is None:
        nodes = quad[:3]
    else:
        nodes = quad
    nquads = " ".join(map(lambda n: canon_ser_node(id_, n), nodes)) + " .\n"
    return nquads


class IdentifierIssuer():
    def __init__(self, prefix='c14n'):
        self.prefix = prefix
        self.counter = 0
        self.issued: OrderedDict[str, str] = OrderedDict()

    def issue(self, identifier):
        if identifier not in self.issued:
            issued_identifier = self.prefix + str(self.counter)
            self.issued[identifier] = issued_identifier
            self.counter += 1
            #print(identifier, "->", self.issued[identifier])
        return self.issued[identifier]

    def copy(self):
        other = IdentifierIssuer(self.prefix)
        other.counter = self.counter
        other.issued = self.issued.copy()
        return other


class PoisonedDatasetException(Exception):
    pass


class TooManyNDegreeCalls(PoisonedDatasetException):
    pass


class TooManyPermutations(PoisonedDatasetException):
    pass


class CanonicalizedGraph:
    def __init__(self, dataset: Dataset, rec_limit=3, perm_limit=8, store="default"):
        # Step 1 from section 4.4.3
        self.bnode_to_quads: OrderedDict[BNode, list[_QuadType]] = OrderedDict()
        self.hash_to_bn_identifiers: OrderedDict[str, list[str]] = {}
        self.dataset = dataset
        self.canon = Dataset()
        self.rec_limit = rec_limit
        self.perm_limit = perm_limit
        self.issued: OrderedDict[str, str] | None = None
        self._canonical_issuer = IdentifierIssuer()
        self._create_canon_state()
        self._compute_first_degree_hashes()
        self._compute_n_degree_hashes()
        self._prepare_canonical_dataset()

    def _create_canon_state(self):
        # Step 2 from section 4.4.3
        for quad in self.dataset:
            for node in quad:
                if not isinstance(node, BNode):
                    continue
                id_ = str(node)
                if id_ not in self.bnode_to_quads:
                    self.bnode_to_quads[id_] = []
                self.bnode_to_quads[id_].append(quad)
        self.issuer = IdentifierIssuer()

    def _compute_first_degree_hashes(self):
        # Step 3 from section 4.4.3
        for n in self.bnode_to_quads.keys():
            hash_ = self._hash_first_degree_quad(n)
            if hash_ not in self.hash_to_bn_identifiers:
                self.hash_to_bn_identifiers[hash_] = []
            self.hash_to_bn_identifiers[hash_].append(str(n))

        # Step 4 from section 4.4.3
        for hash_, ids in sorted(self.hash_to_bn_identifiers.items(), key=lambda x: x[0]):
            if len(ids) > 1:
                continue
            #print(f"Processing {hash_} : {ids[0]}")
            self._canonical_issuer.issue(ids[0])
            del self.hash_to_bn_identifiers[hash_]

    def _hash_first_degree_quad(self, id_):
        quads = self.bnode_to_quads[id_]
        nquads = "".join(sorted(canon_nq_row(id_, quad) for quad in quads))
        hash_ = sha256(nquads.encode('utf-8')).hexdigest()
        #print("hashing", nquads, "=>", hash_)
        return hash_

    def _compute_n_degree_hashes(self):
        # Step 5 from section 4.4.3
        for hash_, ids in sorted(self.hash_to_bn_identifiers.items(), key=lambda x: x[0]):
            assert len(ids) > 1
            hash_path_list = []
            # Note: sorted not required by the spec, done for determinism
            for id_ in sorted(ids):
                if id_ in self._canonical_issuer.issued:
                    continue
                temporary_issuer = IdentifierIssuer('b')
                temporary_issuer.issue(id_)
                result = self._hash_n_degree_quads(id_, temporary_issuer)
                hash_path_list.append(result)
            for issuer, _ in sorted(hash_path_list, key=lambda x: x[1]):
                for id_ in issuer.issued.keys():
                    self._canonical_issuer.issue(id_)

    def _hash_related_blank_node(self, related, quad, issuer, position):
        input_ = position
        if position != 'g':
            input_ += '<' + str(quad[1]) + '>'
        if related in self._canonical_issuer.issued:
            input_ += '_:' + self._canonical_issuer.issued[related]
        elif related in issuer.issued:
            input_ += '_:' + issuer.issued[related]
        else:
            input_ += self._hash_first_degree_quad(related)
        return sha256(input_.encode('utf-8')).hexdigest()

    def _hash_n_degree_quads(self, id_, issuer, level=1):
        #print("n degree level", level)
        if level > self.rec_limit:
            raise TooManyNDegreeCalls()
        # Section 4.8.3
        hash_to_related_bns: dict[str, str] = {}
        quads = self.bnode_to_quads[id_]
        for quad in quads:
            # Note: in rdfc1.0 predicate is ignored as it can't be a blank node in RDF.
            # we also consider it, since in N3 it can be a blank node.
            for c, pos in zip(quad, ['s', 'p', 'o', 'g']):
                rel_id = str(c)
                if isinstance(c, BNode) and rel_id != id_:
                    hash_ = self._hash_related_blank_node(rel_id, quad, issuer, pos)
                    if hash_ not in hash_to_related_bns:
                        hash_to_related_bns[hash_] = []
                    hash_to_related_bns[hash_].append(id_)
        data_to_hash = ""
        for related_hash, bn_list in sorted(hash_to_related_bns.items(), key=lambda x: x[0]):
            data_to_hash += related_hash
            chosen_path = ""
            chosen_issuer = None
            if self.perm_limit > 0 and len(bn_list) > self.perm_limit:
                raise TooManyPermutations()
            for p in permutations(bn_list):
                issuer_copy = issuer.copy()
                path = ""
                recursion_list = []
                for related in p:
                    if related in self._canonical_issuer.issued:
                        path += '_:'
                    else:
                        if related not in issuer_copy.issued:
                            recursion_list.append(related)
                        path += '_:' + issuer_copy.issue(related)
                    if len(chosen_path) > 0 and len(path) > len(chosen_path) and path > chosen_path:
                        continue
                for related in recursion_list:
                    issuer, result = self._hash_n_degree_quads(related, issuer_copy, level+1)
                    path += '_:' + issuer_copy.issue(related)
                    path += '<' + result + '>'
                    issuer_copy = issuer
                    if len(chosen_path) > 0 and len(path) > len(chosen_path) and path > chosen_path:
                        continue
                if len(chosen_path) == 0 or len(path) < len(chosen_path) or path < chosen_path:
                    chosen_path = path
                    chosen_issuer = issuer_copy
            data_to_hash += chosen_path
            issuer = chosen_issuer
        return issuer, sha256(data_to_hash.encode('utf-8')).hexdigest()

    def _prepare_canonical_dataset(self):
        self.issued = self._canonical_issuer.issued
        self.canon = Dataset()
        canon_bnodes = {k: BNode(v) for k, v in self.issued.items()}
        for quad in self.dataset:
            canon_quad = []
            for c_, pos in zip(quad, ['s', 'p', 'o', 'c']):
                if isinstance(c_, BNode):
                    c = canon_bnodes[str(c_)]
                else:
                    c = c_
                if pos == 'c':
                    c = self.canon.graph(c)
                canon_quad.append(c)
            self.canon.add(canon_quad)


def post_canon_cmp(canon: Dataset, cmp: Dataset) -> bool:
    new_nodes: dict[str, BNode] = {}
    i = 0
    for quad in sorted(canon):
        for node in quad:
            if not isinstance(node, BNode):
                continue
            old_id = str(node)
            if old_id not in new_nodes:
                i += 1
                new_nodes[old_id] = BNode(f"bnode_{i}")
    new_canon = Dataset()
    for quad in sorted(canon):
        new_quad = tuple(new_nodes[str(node)] if isinstance(node, BNode) else node for node in quad)
        new_canon.add(new_quad)
    old_ids = list(new_nodes.keys())
    if len(old_ids) > 10:
        return None
    for perm in permutations(new_nodes.values()):
        perm_nodes = dict(zip(old_ids, perm))
        new_cmp = Dataset()
        for quad in sorted(cmp):
            new_quad = tuple(perm_nodes[str(node)] if isinstance(node, BNode) else node for node in quad)
            new_cmp.add(new_quad)
        if set(new_cmp) == set(new_canon):
            return True
    return False
