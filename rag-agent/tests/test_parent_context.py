import copy
import unittest

from src.parent_context import ParentContextIndex


class Tokenizer:
    def __call__(self, query, text, **kwargs):
        return {'input_ids': list(range(len(query) + len(text) + 3))}


class ParentContextTests(unittest.TestCase):
    def setUp(self):
        self.meta = dict(source='synthetic', page=1, block_type='table', block_index=2)
        self.doc = {'text': 'Name | Value | Unit\nAA100 | 20 | MPa\nBB200 | 30 | MPa', 'metadata': self.meta}
        self.child = {'text': '100 | 20 | MPa\nBB200 | 30', 'chunk_id': 1, 'source': 'synthetic', 'metadata': self.meta.copy()}

    def test_deterministic_rows_header_and_provenance(self):
        idx = ParentContextIndex([self.doc]); original = copy.deepcopy(self.child)
        result = idx.expand(self.child, 'q', Tokenizer(), 300)
        self.assertEqual(result, idx.expand(self.child, 'q', Tokenizer(), 300))
        self.assertEqual(result['status'], 'EXPANDED')
        self.assertIn('Name | Value | Unit', result['text'])
        self.assertIn('AA100 | 20 | MPa', result['text'])
        self.assertIn('BB200 | 30 | MPa', result['text'])
        self.assertEqual(result['provenance']['child_id'][-1], 1)
        self.assertEqual(result['provenance']['parent_id'], tuple(self.meta.values()))
        self.assertEqual(self.child, original)

    def test_missing_and_ambiguous_parent(self):
        for docs,status in (([], 'MISSING_PARENT'), ([self.doc,self.doc], 'AMBIGUOUS_PARENT')):
            result=ParentContextIndex(docs).expand(self.child,'q',Tokenizer(),300)
            self.assertEqual(result['status'],status)
            self.assertEqual(result['text'],self.child['text'])

    def test_ambiguous_occurrence_and_missing_child(self):
        for text,child,status in [('same same','same','AMBIGUOUS_CHILD'),('parent','absent','CHILD_NOT_FOUND')]:
            doc=dict(self.doc,text=text); c=dict(self.child,text=child)
            self.assertEqual(ParentContextIndex([doc]).expand(c,'q',Tokenizer(),300)['status'],status)

    def test_cap_falls_back_without_truncating_child(self):
        result=ParentContextIndex([self.doc]).expand(self.child,'q',Tokenizer(),50)
        self.assertEqual(result['status'],'CONTEXT_BUDGET_EXCEEDED')
        self.assertEqual(result['text'],self.child['text'])

    def test_oversized_child_and_incomplete_metadata(self):
        self.assertEqual(ParentContextIndex([self.doc]).expand(self.child,'q',Tokenizer(),10)['status'],'CHILD_OVER_BUDGET')
        c=dict(self.child,metadata={'source':'synthetic'})
        self.assertEqual(ParentContextIndex([self.doc]).expand(c,'q',Tokenizer(),300)['status'],'INCOMPLETE_IDENTITY')

    def test_text_context_is_same_parent_bounded(self):
        meta=dict(self.meta,block_type='text'); doc=dict(text='prefix '+'x'*30+' TARGET '+'y'*30+' suffix',metadata=meta)
        child=dict(self.child,text='TARGET',metadata=meta)
        result=ParentContextIndex([doc]).expand(child,'q',Tokenizer(),300)
        self.assertEqual(result['context_type'],'text_neighborhood')
        self.assertIn('TARGET',result['text'])


if __name__=='__main__':
    unittest.main()
