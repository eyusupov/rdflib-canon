from collections.abc import Iterable
from rdflib import Graph, BNode, Literal
from rdflib.graph import Dataset, _QuadType, DATASET_DEFAULT_GRAPH_ID
from rdflib.plugins.serializers.nt import _quoteLiteral, _nt_row
from rdflib.plugins.serializers.nquads import _nq_row
from hashlib import sha256
from itertools import permutations
from typing import cast
import logging
import json
import yaml

log = logging.getLogger(__name__)


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


def _log_fmt(stage, log_point, data):
    log = {
        stage: {
            "log point": log_point
        }
    }
    log[stage].update(data)
    return log


def _log(stage, log_point, data):
    #logging.debug(yaml.safe_dump(_log_fmt(stage, log_point, data), indent=2))
    logging.debug(json.dumps(_log_fmt(stage, log_point, data), indent=2))


def _log_quad(quad):
    if quad[3] == DATASET_DEFAULT_GRAPH_ID:
        return _nt_row(quad[0:3]).strip()
    else:
        return _nq_row(quad[0:3], quad[3]).strip()


def _log_quads(quads):
    return [_log_quad(quad) for quad in quads]


def _log_bnode_to_quads(bnode_to_quads):
    return {
        id_: _log_quads(quads) for id_, quads in bnode_to_quads.items()
    }


class CanonicalizedGraph:
    def __init__(self, graph: Graph | Dataset, rec_limit=8, perm_limit=8, store="default") -> None:
        # Step 1 from section 4.4.3
        self.bnode_to_quads: dict[str, list[_QuadType]] = {}
        self.hash_to_bn_identifiers: dict[str, list[str]] = {}
        if isinstance(graph, Dataset):
            self.dataset = graph
        else:
            self.dataset = Dataset()
            self.dataset.add_graph(graph)
        self.calls = 0
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
        log.debug("\tlog point: Entering the canonicalization function (4.4.3).")
        for quad in self.dataset:
            for node in quad:
                if not isinstance(node, BNode):
                    continue
                id_ = str(node)
                if id_ not in self.bnode_to_quads:
                    self.bnode_to_quads[id_] = []
                self.bnode_to_quads[id_].append(cast(_QuadType, quad))
        if log.isEnabledFor(logging.DEBUG):
            _log("ca.2", "Extract quads for each bnode (4.4.3 (2)).", {
                "Bnode to quads": _log_bnode_to_quads(self.bnode_to_quads)
            })
        self.issuer = IdentifierIssuer()

    def _compute_first_degree_hashes(self) -> None:
        # Step 3 from section 4.4.3
        h1dq_logs = []
        for id_, quads in self.bnode_to_quads.items():
            h1dq_log = {"identifier": id_}
            hash_ = self._hash_first_degree_quad(id_, quads, h1dq_log)
            if hash_ not in self.hash_to_bn_identifiers:
                self.hash_to_bn_identifiers[hash_] = []
            self.hash_to_bn_identifiers[hash_].append(id_)
            if log.isEnabledFor(logging.DEBUG):
                h1dq_logs.append(h1dq_log)

        if log.isEnabledFor(logging.DEBUG):
            _log("ca.3", "Calculated first degree hashes (4.4.3 (3)).", {
                "with": h1dq_logs
            })

        if log.isEnabledFor(logging.DEBUG):
            _logged_identifiers = []
        # Step 4 from section 4.4.3
        for hash_, ids in sorted(self.hash_to_bn_identifiers.items(), key=lambda x: x[0]):
            if len(ids) > 1:
                continue
            id_, = ids
            self._canonical_issuer.issue(id_)
            if log.isEnabledFor(logging.DEBUG):
                _logged_identifiers.append({
                    "identifier": id_,
                    "hash": hash_,
                    "canonical label": self._canonical_issuer.issued[id_]
                })
            del self.hash_to_bn_identifiers[hash_]

        if log.isEnabledFor(logging.DEBUG):
            _log("ca.4", "Create canonical replacements for hashes mapping to a single node (4.4.3 (4)).", {
                "with": _logged_identifiers
            })

    def _hash_first_degree_quad(self, id_: str, quads: list[_QuadType], log_dict: dict | None = None) -> str:
        nquads = sorted(canon_nq_row(id_, quad) for quad in quads)
        hash_ = sha256("".join(nquads).encode('utf-8')).hexdigest()
        if log.isEnabledFor(logging.DEBUG):
            log_dict.update(_log_fmt("h1dq", "Hash First Degree Quads function (4.6.3).", {
                "nquads": [x.strip() for x in nquads],
                "hash": hash_
            }))
        return hash_

    def _compute_n_degree_hashes(self):
        # Step 5 from section 4.4.3
        if log.isEnabledFor(logging.DEBUG):
            _step_logs = []
        for hash_, ids in sorted(self.hash_to_bn_identifiers.items(), key=lambda x: x[0]):
            assert len(ids) > 1
            hash_path_list = [] # 5.1
            if log.isEnabledFor(logging.DEBUG):
                _step_log = {
                    "hash": hash_,
                    "identifier list": ids,
                }
                hndq_logs = []
            for id_ in ids: # 5.2
                if id_ in self._canonical_issuer.issued:
                    continue # 5.2.1
                temporary_issuer = IdentifierIssuer('b') # 5.2.2
                temporary_issuer.issue(id_) # 5.2.3
                hndq_log = {"identifier": id_}
                hash_path_list.append(self._hash_n_degree_quads(id_, temporary_issuer, log_dict=hndq_log)) # 5.2.4
                if log.isEnabledFor(logging.DEBUG):
                    hndq_logs.append(hndq_log)

            if log.isEnabledFor(logging.DEBUG):
                _step_log.update(_log_fmt("ca.5.2", "Calculate hashes for identifiers with shared hashes (4.4.3 (5.2)).", {
                    "with": hndq_logs
                }))

            _issued_identifiers = []
            for issuer, _ in sorted(hash_path_list, key=lambda x: x[1]): # 5.3
                for id_ in issuer.issued.keys():
                    self._canonical_issuer.issue(id_)
                    _issued_identifiers.append({
                        "blank_node": id_,
                        "canonical identifier": self._canonical_issuer.issued[id_]
                    })
            if log.isEnabledFor(logging.DEBUG):
                _step_log.update(_log_fmt("ca.5.3", "Canonical identifiers for temporary identifiers (4.4.3 (5.3)).", {
                    "with": _issued_identifiers
                }))

                _step_logs.append(_step_log)
                _log("ca.5", "Calculate hashes for identifiers with shared hashes (4.4.3 (5)).", {
                    "with": _step_logs
                })

    def _hash_related_blank_node(self, related: str, quad: _QuadType, issuer: IdentifierIssuer, position: str, log_dict: dict | None = None):
        input_ = position
        _h1dq_log: dict = {}
        if position != 'g':
            input_ += '<' + str(quad[1]) + '>'
        if related in self._canonical_issuer.issued:
            input_ += '_:' + self._canonical_issuer.issued[related]
        elif related in issuer.issued:
            input_ += '_:' + issuer.issued[related]
        else:
            input_ += self._hash_first_degree_quad(related, self.bnode_to_quads[related], _h1dq_log)
        hash_ = sha256(input_.encode('utf-8')).hexdigest()
        if log.isEnabledFor(logging.DEBUG):
            assert log_dict is not None
            log_dict.update(_log_fmt("hndq3.1", "Hash related bnode component (4.8.3 (3.1))",
                 {
                     "with": {
                         "position": position,
                         "related": related,
                         "h1dq": _h1dq_log,
                         "input": input_,
                         "hash": hash_
                     }
                 }))

        return hash_


    def _hash_n_degree_quads(self, id_: str, issuer: IdentifierIssuer, level=1, log_dict: dict | None =None) -> tuple[IdentifierIssuer, str]:
        self.calls += 1
        if level > self.rec_limit:
            raise TooManyNDegreeCalls()
        # Section 4.8.3
        hash_to_related_bns: dict[str, list[str]] = {} # Step 1
        quads = self.bnode_to_quads[id_] # Step 2
        if log.isEnabledFor(logging.DEBUG):
            assert log_dict is not None
            log_dict['hndq'] = {
                'identifier': id_,
                'log point': 'Hash N-Degree Quads function (4.8.3).',
                'issuer': issuer.issued,
                'hndq.2': {
                    'log point': 'Quads for identifier (4.8.3 (2)).',
                    'quads': _log_quads(quads)
                },
            }

        _log_list = []
        for quad in quads: # Step 3
            if log.isEnabledFor(logging.DEBUG):
                _with_list = []
                _quad_dict = {
                    "quad": _log_quad(quad),
                    "hndq.3.1": {
                        "log point": "Hash related bnode component (4.8.3 (3.1))",
                        'with': _with_list
                    }
                }
            # Note: in rdfc1.0 predicate is ignored as it can't be a blank node in RDF.
            # we also consider it, since in N3 it can be a blank node.
            for c, pos in zip(quad, ['s', 'p', 'o', 'g']):
                rel_id = str(c)
                if isinstance(c, BNode) and rel_id != id_: # 3.1
                    _hrbn_dict: dict = {}
                    hash_ = self._hash_related_blank_node(rel_id, quad, issuer, pos, _hrbn_dict) # 3.1.1
                    if hash_ not in hash_to_related_bns: # 3.1.2
                        hash_to_related_bns[hash_] = []
                    hash_to_related_bns[hash_].append(rel_id)
                    if log.isEnabledFor(logging.DEBUG):
                        _with_list.append(_hrbn_dict)
            if log.isEnabledFor(logging.DEBUG):
                _log_list.append(_quad_dict)

        if log.isEnabledFor(logging.DEBUG):
            assert log_dict is not None
            log_dict['hndq']['hndq.3'] = {
                'log point': 'Hash N-Degree Quads function (4.8.3 (3)).',
                'with': _log_list,
                'Hash to bnodes': hash_to_related_bns
            }
        data_to_hash = "" # Step 4

        if log.isEnabledFor(logging.DEBUG):
            assert log_dict is not None
            log_dict['hndq'].update(_log_fmt("hndq.5", "Hash N-Degree Quads function (4.8.3 (5)), entering loop.", {
                "with": []
            }))

        for related_hash, bn_list in sorted(hash_to_related_bns.items(), key=lambda x: x[0]): # Step 5
            if log.isEnabledFor(logging.DEBUG):
                assert log_dict is not None
                log_dict['hndq']['hndq.5']['with'].append({
                    "related_hash": related_hash,
                    "data_to_hash": data_to_hash
                })
            data_to_hash += related_hash # 5.1
            chosen_path = "" # 5.2
            chosen_issuer = None # 5.3
            if self.perm_limit > 0 and len(bn_list) > self.perm_limit:
                raise TooManyPermutations()
            for permutation in permutations(bn_list): # 5.4
                if log.isEnabledFor(logging.DEBUG):
                    assert log_dict is not None
                    log_perm = {'perm': permutation}
                    log_dict['hndq']['hndq.5']['with'].append(log_perm)
                    log_perm_list: list = []
                    log_perm.update(_log_fmt('hndq.5.4.4', 'Hash N-Degree Quads function (4.8.3 (5.4.4)), entering loop.', {'with': log_perm_list}))
                issuer_copy = issuer.copy() # 5.4.1
                path = "" # 5.4.2
                recursion_list = [] # 5.4.3
                for related in permutation: # 5.4.4
                    if log.isEnabledFor(logging.DEBUG):
                        log_perm_list.append({
                            'related': related,
                            'path': path
                        })

                    if related in self._canonical_issuer.issued:
                        path += '_:' + self._canonical_issuer.issued[related] # 5.4.4.1
                    else:
                        if related not in issuer_copy.issued:
                            recursion_list.append(related) # 5.4.4.2.1
                        path += '_:' + issuer_copy.issue(related) # 5.4.4.2.2
                    if len(chosen_path) > 0 and len(path) >= len(chosen_path) and path > chosen_path: # 5.4.4.3
                       break

                if log.isEnabledFor(logging.DEBUG):
                    log_list: list = []
                    log_perm.update(_log_fmt('hndq.5.4.5', 'Hash N-Degree Quads function (4.8.3 (5.4.5)), before possible recursion.', {
                        'recursion_list': recursion_list,
                        'path': path,
                        'with': log_list
                    }))
                for related in recursion_list: # 5.4.5
                    hndq_log: dict = {}
                    issuer_, result = self._hash_n_degree_quads(related, issuer_copy, level+1, log_dict=hndq_log) # 5.4.5.1
                    if log.isEnabledFor(logging.DEBUG):
                        hndq_log['related'] = related
                        log_list.append(hndq_log)
                    path += '_:' + issuer_copy.issue(related) # 5.4.5.2
                    path += '<' + result + '>' # 5.4.5.3
                    issuer_copy = issuer_ # 5.4.5.4
                    if len(chosen_path) > 0 and len(path) >= len(chosen_path) and path > chosen_path: # 5.4.5.5
                        break
                if len(chosen_path) == 0 or path < chosen_path: # 5.4.5.6
                    chosen_path = path
                    chosen_issuer = issuer_copy
            data_to_hash += chosen_path # 5.5
            assert chosen_issuer
            issuer = chosen_issuer
            if log.isEnabledFor(logging.DEBUG):
                assert log_dict is not None
                log_dict['hndq']['hndq.5'].update(_log_fmt("hndq5.5", "Hash N-Degree Quads function (4.8.3 (5.5). End of current loop with Hn hashes.", {
                    "chosen path": chosen_path,
                    "data to hash": data_to_hash
                }))

        hash_ = sha256(data_to_hash.encode('utf-8')).hexdigest() # Step 6
        if log.isEnabledFor(logging.DEBUG):
            assert log_dict is not None
            log_dict['hndq'].update(_log_fmt('hndq.6', 'Leaving Hash N-Degree Quads function (4.8.3).', {
                'hash': hash_,
                'issuer': issuer.issued
            }))
        return issuer, hash_

    def _prepare_canonical_dataset(self):
        self.issued = self._canonical_issuer.issued
        _log("ca.6", "Issued identifiers map (4.4.3 (6)).", {
            "issued identifiers map: ": self._canonical_issuer.issued
        })
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
