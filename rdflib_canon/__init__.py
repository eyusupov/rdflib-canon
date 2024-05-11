from collections.abc import Iterable
from rdflib import Graph, BNode, Literal
from rdflib.graph import Dataset, _QuadType, DATASET_DEFAULT_GRAPH_ID
from rdflib.plugins.serializers.nt import _quoteLiteral
from hashlib import sha256
from itertools import permutations
from typing import cast


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
    def __init__(self, prefix='c14n') -> None:
        self.prefix = prefix
        self.counter = 0
        self.issued: dict[str, str] = {}

    def issue(self, identifier) -> str:
        if identifier not in self.issued:
            issued_identifier = self.prefix + str(self.counter)
            self.issued[identifier] = issued_identifier
            self.counter += 1
        return self.issued[identifier]

    def copy(self) -> "IdentifierIssuer":
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
    def __init__(self, graph: Graph | Dataset, rec_limit=3, perm_limit=8, store="default") -> None:
        # Step 1 from section 4.4.3
        self.bnode_to_quads: dict[str, list[_QuadType]] = {}
        self.hash_to_bn_identifiers: dict[str, list[str]] = {}
        if isinstance(graph, Dataset):
            self.dataset = graph
        else:
            self.dataset = Dataset()
            self.dataset.add_graph(graph)
        self.rec_limit = rec_limit
        self.perm_limit = perm_limit
        self.issued: dict[str, str] | None = None
        self._canonical_issuer = IdentifierIssuer()
        self._create_canon_state()
        self._compute_first_degree_hashes()
        self._compute_n_degree_hashes()
        self._prepare_canonical_dataset()

    def _create_canon_state(self) -> None:
        # Step 2 from section 4.4.3
        for quad in self.dataset:
            for node in quad:
                if not isinstance(node, BNode):
                    continue
                id_ = str(node)
                if id_ not in self.bnode_to_quads:
                    self.bnode_to_quads[id_] = []
                self.bnode_to_quads[id_].append(cast(_QuadType, quad))
        self.issuer = IdentifierIssuer()

    def _compute_first_degree_hashes(self) -> None:
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
            self._canonical_issuer.issue(ids[0])
            del self.hash_to_bn_identifiers[hash_]

    def _hash_first_degree_quad(self, id_: str) -> str:
        quads = self.bnode_to_quads[id_]
        nquads = "".join(sorted(canon_nq_row(id_, quad) for quad in quads))
        hash_ = sha256(nquads.encode('utf-8')).hexdigest()
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

    def _hash_related_blank_node(self, related: str, quad: _QuadType, issuer: IdentifierIssuer, position: str):
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

    def _hash_n_degree_quads(self, id_: str, issuer: IdentifierIssuer, level=1) -> tuple[IdentifierIssuer, str]:
        if level > self.rec_limit:
            raise TooManyNDegreeCalls()
        # Section 4.8.3
        hash_to_related_bns: dict[str, list[str]] = {}
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
            assert chosen_issuer
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


def permute_ds_bnode_ids(ds: Dataset) -> Iterable[dict[BNode, BNode]]:
    bnodes = [bnode for quad in ds for bnode in quad if isinstance(bnode, BNode)]
    if len(bnodes) > 10:
        raise TooManyPermutations
    for perm in permutations(bnodes):
        yield dict(zip(bnodes, perm))

def permute_ds_bnodes(ds: Dataset) -> Iterable[Dataset]:
    for bnode_map in permute_ds_bnode_ids(ds):
        new_ds = Dataset()
        for quad in sorted(ds):
            new_quad = tuple(bnode_map[node] if isinstance(node, BNode) else node for node in quad)
            new_ds.add(cast(_QuadType, new_quad))
        yield new_ds
