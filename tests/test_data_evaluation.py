"""Meaningful corpus-isolation and metric-boundary regression tests; stdlib only."""
import copy
import hashlib
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded

E = module('mc_evaluation', ROOT / 'evaluation/evaluate.py')
G = module('mc_generator', ROOT / 'data/generate.py')


def gold(identifier, decision, skill=None, family='family_a'):
    return {'id': identifier, 'family': family, 'output': json.dumps({'decision': decision, 'skill': skill})}


def prediction(identifier, decision, skill=None):
    return {'id': identifier, 'raw_output': json.dumps({'decision': decision, 'skill': skill})}


class EvaluationTests(unittest.TestCase):
    def test_wrong_skill_keeps_decision_accuracy_but_fails_joint(self):
        result = E.evaluate([gold('a', 'replace_task', 'collect')], [prediction('a', 'replace_task', 'craft')])
        self.assertEqual(result['decision_accuracy'], 1)
        self.assertEqual(result['joint_accuracy'], 0)
        self.assertEqual(result['skill_accuracy'], 0)
        self.assertEqual(result['strict_json_validity'], 1)

    def test_non_json_and_schema_errors_are_not_salvaged(self):
        bad = ['```json\n{"decision":"cancel","skill":null}\n```',
               '{"decision":"cancel","skill":null} done',
               '{"decision":"cancel","decision":"continue","skill":null}',
               '{"decision":"cancel","skill":"collect"}',
               '{"decision":"replace_task","skill":null}',
               '{"decision":"continue","skill":null,"reason":"ok"}',
               '{"decision":["cancel"],"skill":null}']
        for raw in bad:
            with self.subTest(raw=raw):
                self.assertIsNotNone(E.parse_output(raw)[1])
        result = E.evaluate([gold('a', 'cancel')], [{'id': 'a', 'raw_output': bad[0]}])
        self.assertEqual(result['strict_json_validity'], 0)
        self.assertEqual(result['missed_cancel_rate'], {'value': 1, 'numerator': 1, 'denominator': 1})

    def test_missing_ids_are_errors_in_denominators(self):
        expected = [gold('a', 'cancel'), gold('b', 'continue')]
        result = E.evaluate(expected, [prediction('b', 'continue')])
        self.assertEqual(result['n'], 2)
        self.assertEqual(result['joint_accuracy'], .5)
        self.assertEqual(result['invalid_or_missing']['missing_prediction'], 1)
        self.assertEqual(result['missed_cancel_rate']['denominator'], 1)

    def test_unknown_and_duplicate_ids_fail(self):
        expected = [gold('a', 'continue')]
        for predictions in [[prediction('b', 'continue')], [prediction('a', 'continue')] * 2]:
            with self.assertRaises(ValueError):
                E.evaluate(expected, predictions)

    def test_prediction_order_does_not_change_matching(self):
        expected = [gold('a', 'cancel'), gold('b', 'continue')]
        result = E.evaluate(expected, [prediction('b', 'continue'), prediction('a', 'cancel')])
        self.assertEqual(result['joint_accuracy'], 1)

    def test_raw_output_authoritative_and_coverage_explicit(self):
        expected = [gold('a', 'continue')]
        pred = {'id': 'a', 'raw_output': 'bad', 'prediction': {'decision': 'continue', 'skill': None}}
        self.assertEqual(E.evaluate(expected, [pred])['joint_accuracy'], 0)
        del pred['raw_output']
        result = E.evaluate(expected, [pred])
        self.assertEqual(result['joint_accuracy'], 1)
        self.assertEqual(result['strict_json_validity'], 0)
        self.assertEqual(result['raw_output_coverage']['numerator'], 0)

    def test_cancel_denominators_and_skill_denominator(self):
        expected = [gold('a', 'continue'), gold('b', 'pause'), gold('c', 'cancel'), gold('d', 'replace_task', 'craft')]
        predicted = [prediction('a', 'cancel'), prediction('b', 'pause'), prediction('c', 'continue'), prediction('d', 'pause')]
        result = E.evaluate(expected, predicted)
        self.assertEqual(result['false_cancel_rate'], {'value': 1/3, 'numerator': 1, 'denominator': 3})
        self.assertEqual(result['missed_cancel_rate']['value'], 1)
        self.assertEqual(result['skill_accuracy_counts']['denominator'], 1)
        self.assertEqual(result['skill_accuracy'], 0)

    def test_macro_f1_counts_invalid_false_negatives(self):
        expected = [gold(str(i), decision, 'collect' if decision == 'replace_task' else None) for i, decision in enumerate(E.DECISIONS)]
        predicted = [prediction(str(i), decision, 'collect' if decision == 'replace_task' else None) for i, decision in enumerate(E.DECISIONS)]
        predicted[0]['raw_output'] = 'bad'
        result = E.evaluate(expected, predicted)
        self.assertEqual(result['macro_f1'], .8)
        self.assertEqual(result['confusion']['continue']['invalid'], 1)

    def test_family_bootstrap_identity_and_direction(self):
        expected = [gold('a', 'continue', family='x'), gold('b', 'cancel', family='y')]
        good = E.align(expected, [prediction('a', 'continue'), prediction('b', 'cancel')])
        bad = E.align(expected, [prediction('a', 'pause'), prediction('b', 'pause')])
        result = E.paired_family_bootstrap(bad, good, iterations=20)
        self.assertEqual(result['metrics']['joint_accuracy']['difference'], 1)
        self.assertEqual(result['metrics']['joint_accuracy']['percentile_95_ci'], [1, 1])
        same = E.paired_family_bootstrap(good, good, iterations=20)
        self.assertEqual(same['metrics']['macro_f1']['percentile_95_ci'], [0, 0])

    def test_latency_alias_and_invalid_latency(self):
        expected = [gold('a', 'continue')]
        pred = prediction('a', 'continue'); pred['latency'] = .5
        self.assertEqual(E.evaluate(expected, [pred])['latency_seconds']['mean'], .5)
        pred['latency'] = -1
        with self.assertRaises(ValueError):
            E.evaluate(expected, [pred])


class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.datasets, cls.registry = G.build()

    def test_frozen_content_counts_balance_and_no_family_input_leakage(self):
        stats = G.validate(self.datasets)
        self.assertEqual([stats[s]['examples'] for s in ('train', 'val', 'test')], [1000, 160, 300])
        manifest = json.loads((ROOT / 'data/manifest.json').read_text())
        for split, rows in self.datasets.items():
            path = ROOT / 'data' / f'{split}.json'
            self.assertEqual(json.loads(path.read_text()), rows)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), manifest['sha256'][path.name])

    def test_cross_split_family_leakage_rejected(self):
        bad = copy.deepcopy(self.datasets)
        bad['val'][0]['family'] = bad['train'][0]['family']
        with self.assertRaisesRegex(AssertionError, 'family leakage'):
            G.validate(bad)

    def test_exact_input_leakage_rejected(self):
        bad = copy.deepcopy(self.datasets)
        bad['val'][0]['input'] = bad['train'][0]['input']
        with self.assertRaisesRegex(AssertionError, 'duplicate rendered input'):
            G.validate(bad)

    def test_challenge_contains_contextual_minimal_sets(self):
        challenge = [r for r in self.datasets['test'] if r['family'].startswith('challenge_')]
        self.assertEqual(len(challenge), 100)
        bell = [r for r in challenge if r['family'] == 'challenge_coded_bell']
        self.assertEqual(len({json.loads(r['input'])['latest_utterance'] for r in bell}), 1)
        self.assertEqual({json.loads(r['output'])['decision'] for r in bell}, set(E.DECISIONS))
        self.assertEqual(len({r['input'] for r in bell}), 5)

if __name__ == '__main__':
    unittest.main()
