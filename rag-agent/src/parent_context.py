"""Offline, QA-blind bounded context recovery from unchanged Documents.

This module does not retrieve, tokenize for BM25, alter chunks, or select answers.
Serialized table prefixes are retained as context, not assumed semantic headers.
"""

KEYS = ('source', 'page', 'block_type', 'block_index')


def parent_identity(value):
    metadata = value.get('metadata') or {}
    identity = tuple(metadata.get(key) for key in KEYS)
    return identity if all(part is not None for part in identity) else None


def pair_tokens(tokenizer, query, text):
    return len(tokenizer(query, text, truncation=False, add_special_tokens=True)['input_ids'])


class ParentContextIndex:
    def __init__(self, documents):
        self.parents = {}
        for document in documents:
            identity = parent_identity(document)
            if identity is not None:
                self.parents.setdefault(identity, []).append(document)

    def expand(self, child, query, tokenizer, max_pair_tokens=768):
        if max_pair_tokens <= 0:
            raise ValueError('Token budget must be positive')
        text = child['text']
        identity = parent_identity(child)
        provenance = {'parent_id': identity,
                      'child_id': (*identity, child.get('chunk_id')) if identity else None,
                      'source': (child.get('metadata') or {}).get('source'),
                      'page': (child.get('metadata') or {}).get('page'),
                      'block_type': (child.get('metadata') or {}).get('block_type'),
                      'block_index': (child.get('metadata') or {}).get('block_index')}
        result = {'text': text, 'context_type': 'child_only', 'provenance': provenance,
                  'pair_tokens': pair_tokens(tokenizer, query, text), 'row_indices': []}

        def fallback(status):
            return dict(result, status=status)

        if result['pair_tokens'] > max_pair_tokens:
            return fallback('CHILD_OVER_BUDGET')
        if identity is None:
            return fallback('INCOMPLETE_IDENTITY')
        parents = self.parents.get(identity, [])
        if not parents:
            return fallback('MISSING_PARENT')
        if len(parents) != 1:
            return fallback('AMBIGUOUS_PARENT')
        parent = parents[0]['text']
        start = parent.find(text)
        if start < 0 or not text:
            return fallback('CHILD_NOT_FOUND')
        if parent.find(text, start + 1) >= 0:
            return fallback('AMBIGUOUS_CHILD')
        end = start + len(text)
        if identity[2] == 'table':
            # Complete only clipped boundary rows; interior rows are already in child.
            rows = parent.splitlines(keepends=True)
            offset = 0
            boundaries = []
            for i, row in enumerate(rows):
                row_end = offset + len(row.rstrip('\r\n'))
                if offset < end and row_end > start and (offset < start or row_end > end):
                    boundaries.append(i)
                offset += len(row)
            leading = '\n'.join(parent.splitlines()[:3])
            recovered = '\n'.join(rows[i].rstrip('\r\n') for i in boundaries)
            expanded = '[Table leading context]\n' + leading
            if recovered:
                expanded += '\n\n[Complete boundary rows]\n' + recovered
            expanded += '\n\n[Retrieved child]\n' + text
            context_type = 'table_boundary_rows' if boundaries else 'table_prefix'
        else:
            neighborhood = parent[max(0, start - 120):min(len(parent), end + 120)]
            expanded = '[Parent neighborhood]\n' + neighborhood + '\n\n[Retrieved child]\n' + text
            boundaries = []
            context_type = 'text_neighborhood'
        length = pair_tokens(tokenizer, query, expanded)
        if length > max_pair_tokens:
            return fallback('CONTEXT_BUDGET_EXCEEDED')
        return dict(result, text=expanded, status='EXPANDED', context_type=context_type,
                    pair_tokens=length, row_indices=boundaries)
