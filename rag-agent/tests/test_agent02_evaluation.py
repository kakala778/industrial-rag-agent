import unittest


class Agent02EvaluationTests(unittest.TestCase):
    def test_fidelity_preserves_numeric_units_conditions_and_comparators(self):
        from evaluation.agent02_metrics import quote_fidelity
        self.assertEqual(quote_fidelity('25 mm2', 'wire: 25 mm2 copper'), 'EXACT')
        self.assertEqual(quote_fidelity('25 mm²', r'wire: 25\,mm^{2} copper'), 'NORMALIZED_EQUIVALENT')
        for quote,text in [('25 mm2','250 mm2'), ('5 m','5 mm'), ('at least 5 m','at most 5 m'),
                           ('≥5m','≤5m'), ('5M','5m'), ('25mm2','2x25mm2')]:
            with self.subTest(quote=quote,text=text):
                self.assertEqual(quote_fidelity(quote,text),'NOT_FAITHFUL')

    def test_id_only_rendering_requires_observed_looked_up_source_and_unique_id(self):
        from evaluation.agent02_metrics import render_selected_ids
        looked={'ev_a':dict(evidence_id='ev_a',source='A',text='authentic text',page=1)}
        selection=[dict(scope='A',evidence_id='ev_a')]
        self.assertEqual(render_selected_ids(selection,['ev_a'],looked)[0]['quote'],'authentic text')
        for rows,observed in [(selection,[]),([dict(scope='B',evidence_id='ev_a')],['ev_a']),
                               (selection+selection,['ev_a']),([dict(scope='A',evidence_id='invented')],['ev_a'])]:
            with self.assertRaises(ValueError): render_selected_ids(rows,observed,looked)

    def test_retrieval_bound_and_preflight_are_not_selection_denominators(self):
        from evaluation.agent02_metrics import candidate_group
        oracle={'A':dict(expected_available=True),'B':dict(expected_available=True)}
        reviews={'one':dict(scope='A',label='RELEVANT'),'two':dict(scope='B',label='PARTIAL')}
        self.assertEqual(candidate_group(oracle,reviews),'RETRIEVAL_BOUND')
        reviews['two']['label']='RELEVANT'
        self.assertEqual(candidate_group(oracle,reviews),'CANDIDATE_AVAILABLE')
        self.assertEqual(candidate_group(None,{}),'PREFLIGHT_CONTROL')
        self.assertEqual(candidate_group({'A':dict(expected_available=None)},{}),'UNASSESSED')

    def test_correct_ids_but_wrong_quote_are_separate_and_id_rendering_is_offline(self):
        from evaluation.agent02_metrics import score_selection
        state=dict(looked_up_evidence={'ev_a':dict(source='A',text='25 mm2',page=1)},
                   evidence_ids=['ev_a'],status='invalid_action',findings=[])
        action=dict(action='FINISH',findings=[dict(scope='A',evidence_id='ev_a',quote='250 mm2')])
        oracle={'A':dict(expected_available=True)}
        review={'ev_a':dict(scope='A',label='RELEVANT',field_alignment=True,unit_alignment=True,
                             condition_alignment=None,scope_alignment=True)}
        score=score_selection(state,[action],oracle,review)
        self.assertTrue(score['id_selection_success'])
        self.assertFalse(score['fully_relevant_task_success'])
        self.assertEqual(score['quote_fidelity'],['NOT_FAITHFUL'])
        self.assertTrue(score['host_rendered_selection_success'])
        self.assertEqual(state['findings'],[])

    def test_quote_omission_of_condition_is_partial_even_with_correct_id(self):
        from evaluation.agent02_metrics import score_selection
        state=dict(looked_up_evidence={'ev_a':dict(source='A',text='if indoors wire: 25 mm2',page=1)},
                   evidence_ids=['ev_a'],status='finished',findings=[dict(scope='A',evidence_id='ev_a',quote='25 mm2')])
        action=dict(action='FINISH',findings=state['findings'])
        oracle={'A':dict(reviewed=True,expected_available=True,pages=[1],field_groups=[['25']],
                        unit_groups=[['mm2']],condition_groups=[['indoors']])}
        review={'ev_a':dict(scope='A',label='RELEVANT')}
        score=score_selection(state,[action],oracle,review)
        self.assertTrue(score['id_selection_success'])
        self.assertEqual(score['quote_fidelity'],['EXACT'])
        self.assertFalse(score['fully_relevant_task_success'])

    def test_fidelity_does_not_match_value_inside_larger_product(self):
        from evaluation.agent02_metrics import quote_fidelity
        for q,t in [('5m','15m'),('25mm2','125mm2'),('5.0m','15.0m')]:
            self.assertEqual(quote_fidelity(q,t),'NOT_FAITHFUL')

    def test_format_normalization_does_not_change_numeric_exponents(self):
        from evaluation.agent02_metrics import quote_fidelity
        for q,t in [('103 m',r'10^{3} m'),('23 m',r'2^{3} m'),('103m','10³m')]:
            self.assertEqual(quote_fidelity(q,t),'NOT_FAITHFUL')

    def test_unit_exponents_require_complete_square_or_cube(self):
        from evaluation.agent02_metrics import quote_fidelity
        for q,t in [('5 m20',r'5 m^{20}'),('25 mm23',r'25 mm^{23}'),
                    ('5 m23',r'5 m^23'),('5 m23',r'5 m^{2}3')]:
            self.assertEqual(quote_fidelity(q,t),'NOT_FAITHFUL')
        self.assertEqual(quote_fidelity('5 m2',r'5 m^{2}'),'NORMALIZED_EQUIVALENT')

    def test_explicit_numeric_multiplication_formatting_is_equivalent(self):
        from evaluation.agent02_metrics import quote_fidelity
        self.assertEqual(quote_fidelity('30mm*3mm',r'30mm \times 3mm'),'NORMALIZED_EQUIVALENT')

    def test_manual_review_requires_finding_fingerprint(self):
        from evaluation.agent02_metrics import score_selection
        action=dict(action='FINISH',findings=[dict(scope='A',evidence_id='ev_a',quote='q')])
        state=dict(status='finished',evidence_ids=['ev_a'],looked_up_evidence={'ev_a':dict(source='A',text='q')})
        with self.assertRaisesRegex(ValueError,'stale'):
            score_selection(state,[action],None,{},finding_reviews=[dict(finding_sha256='wrong',label='RELEVANT')])
