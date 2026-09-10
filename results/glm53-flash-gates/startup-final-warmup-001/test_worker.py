"""Actual worker startup method; no CUDA load required for the RED/contract test."""
import contextlib,importlib.util,io,json,os,sys,types,unittest
from pathlib import Path
from unittest import mock
SOURCE=Path('/home/bmarti44/spark-deepseek-v4-flash/scripts/lib/glm53_worker.py')
FLAG='GLM53_RELEASE_WARMUP_CACHE'
class FinalWarmupTests(unittest.TestCase):
    def load(self,setting=None,parent_error=False):
        events=[];state={'reserved':1342177280};result=object()
        class Parent:
            def compile_or_warm_up_model(self):
                events.append('warmup')
                if parent_error:raise RuntimeError('parent warmup failed')
                return result
        def reserved():events.append('reserved');return state['reserved']
        def collect():events.append('gc')
        def release():events.append('release');state['reserved']=0
        modules={'torch':types.SimpleNamespace(cuda=types.SimpleNamespace(memory_reserved=reserved),accelerator=types.SimpleNamespace(empty_cache=release)), 'vllm.v1.worker.gpu_worker':types.SimpleNamespace(Worker=Parent)}
        env=dict(os.environ);env.pop(FLAG,None)
        if setting is not None:env[FLAG]=setting
        with mock.patch.dict(sys.modules,modules),mock.patch.dict(os.environ,env,clear=True):
            spec=importlib.util.spec_from_file_location('final_warmup_test',SOURCE);api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
        api.gc=types.SimpleNamespace(collect=collect)
        return api,Parent,events,state,result
    def test_enabled_releases_unused_warmup_allocation_before_return(self):
        api,parent,events,state,result=self.load('1');output=io.StringIO()
        with contextlib.redirect_stdout(output):got=api.WarmupCleanupWorker().compile_or_warm_up_model()
        self.assertIs(got,result)
        self.assertEqual(state['reserved'],0,'unused warmup allocation still retained after warmup')
        self.assertEqual(events,['warmup','reserved','gc','release','reserved'])
        self.assertEqual(json.loads(output.getvalue()),{'event':'glm53_release_warmup_cache','reserved_before_bytes':1342177280,'reserved_after_bytes':0})
    def test_disabled_inherits_parent_method_exactly(self):
        for setting in [None,'0','true','01']:
            with self.subTest(setting=setting):
                api,parent,events,state,result=self.load(setting)
                self.assertIs(api.WarmupCleanupWorker.compile_or_warm_up_model,parent.compile_or_warm_up_model)
                self.assertIs(api.WarmupCleanupWorker().compile_or_warm_up_model(),result)
                self.assertEqual(events,['warmup'])
    def test_initialization_choice_does_not_reread_environment(self):
        for initial,later,want in [('1','0',0),('0','1',1342177280)]:
            api,parent,events,state,result=self.load(initial)
            with mock.patch.dict(os.environ,{FLAG:later}),contextlib.redirect_stdout(io.StringIO()):
                self.assertIs(api.WarmupCleanupWorker().compile_or_warm_up_model(),result)
            self.assertEqual(state['reserved'],want)
    def test_parent_failure_propagates_without_cleanup_or_replacement(self):
        api,parent,events,state,result=self.load('1',parent_error=True)
        with self.assertRaisesRegex(RuntimeError,'parent warmup failed'):api.WarmupCleanupWorker().compile_or_warm_up_model()
        self.assertEqual(events,['warmup'])
if __name__=='__main__':unittest.main()
