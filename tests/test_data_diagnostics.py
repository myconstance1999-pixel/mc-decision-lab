import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'evaluation'))
from diagnostics import extract_one_object,score

class DiagnosticTests(unittest.TestCase):
    def test_no_null_repair(self):
        gold=[{'id':'a','family':'x','output':json.dumps({'decision':'continue','skill':None})}]
        pred=[{'id':'a','raw_output':'{"decision":"continue","skill":"collect"}'}]
        result=score(gold,pred)
        self.assertEqual(result['decision_accuracy']['value'],1)
        self.assertEqual(result['joint_accuracy']['value'],0)
        self.assertEqual(result['schema_validity']['value'],0)
        self.assertEqual(result['raw_json_syntax_validity']['value'],1)
    def test_single_object_can_be_extracted_without_changing_values(self):
        raw='Here is it: ```json\n{"decision":"pause","skill":null}\n```'
        self.assertEqual(extract_one_object(raw),{'decision':'pause','skill':None})
    def test_ambiguous_or_broken_outer_is_not_repaired(self):
        for raw in ['{"decision":"pause","decision":"cancel","skill":null}',
                    '{} {}','bad { malformed {"decision":"cancel","skill":null}}',
                    '[{"decision":"cancel","skill":null}]']:
            with self.subTest(raw=raw):self.assertIsNone(extract_one_object(raw))
    def test_missing_prediction_keeps_denominator(self):
        gold=[{'id':'a','family':'x','output':'{"decision":"continue","skill":null}'}]
        self.assertEqual(score(gold,[])['decision_accuracy'],{'value':0,'numerator':0,'denominator':1})

if __name__=='__main__':unittest.main()
