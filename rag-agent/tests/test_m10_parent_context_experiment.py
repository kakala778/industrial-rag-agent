import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from evaluation.run_m10_parent_context_experiment import rerank_original_children, replay_pool, safe_output_root, mechanism_passed


class ExperimentTests(unittest.TestCase):
    def test_rank_gain_without_target_restoration_does_not_pass(self):
        target={'baseline_rank':7,'parent_rank':1,'mapping_status':'CONTEXT_BUDGET_EXCEEDED',
                'context_type':'child_only','row_indices':[]}
        row={'candidates':[{}]*13+[target]}
        self.assertFalse(mechanism_passed(row))
        target.update(mapping_status='EXPANDED',context_type='table_boundary_rows',row_indices=[10])
        self.assertTrue(mechanism_passed(row))
        target['parent_rank']=7
        self.assertFalse(mechanism_passed(row))

    def test_expanded_text_is_only_model_input(self):
        children=[{'text':'original', 'source':'synthetic', 'chunk_id':0,
                   'metadata':{'source':'synthetic','page':1,'block_type':'table','block_index':0}}]
        before=copy.deepcopy(children)
        with patch('evaluation.run_m10_parent_context_experiment.rerank',return_value=[dict(children[0],text='expanded',reranker_score=.8)]):
            ranked=rerank_original_children('query',children,[{'text':'expanded'}],object())
        self.assertEqual(ranked[0]['text'],'original')
        self.assertEqual(ranked[0]['reranker_score'],.8)
        self.assertEqual(children,before)

    def test_replay_checks_identity_and_duplicates(self):
        child={'text':'private', 'source':'synthetic','chunk_id':1,'metadata':{'source':'synthetic','page':1,'block_type':'table','block_index':2}}
        row={'corpus_index':0,'chunk_id':1,'metadata':{'source_id':'S001','page':1,'block_type':'table','block_index':2},'score':.1}
        self.assertEqual(replay_pool([row],[child],{'synthetic':'S001'})[0]['text'],'private')
        with self.assertRaises(ValueError): replay_pool([dict(row,chunk_id=2)],[child],{'synthetic':'S001'})
        with self.assertRaises(ValueError): replay_pool([row,row],[child],{'synthetic':'S001'})

    def test_output_must_be_ignored_and_in_designated_root(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError): safe_output_root(Path(temp))


if __name__=='__main__': unittest.main()
