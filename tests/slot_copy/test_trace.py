import contextlib,io,os,threading,types,unittest
from unittest import mock
from serve.server import StrataEngine
class Trace(unittest.TestCase):
    def engine(self,on):
        e=object.__new__(StrataEngine);e.slot_copy_trace=on;e.slot_trace_local=threading.local()
        e.slot_trace_lock=threading.Lock();e.slot_trace_request=0;e.slot_trace_legs={};e.wlock=threading.Lock()
        e.proc=types.SimpleNamespace(pid=123,stdin=io.StringIO());return e
    def test_legs_and_requests_no_prompt_logging(self):
        e=self.engine(True);out=io.StringIO()
        with contextlib.redirect_stderr(out):
            e._slot_trace_begin();e._send('GEN 2 PRIVATE_PROMPT');e._slot_trace_first();e._slot_trace_first()
            e._send('STOP');e._send('BGEN 0 2 PRIVATE_PROMPT');e._slot_trace_begin();e._send('GEN 2 PRIVATE_PROMPT')
        s=out.getvalue();self.assertNotIn('PRIVATE_PROMPT',s);self.assertEqual(s.count('event=first_engine_token'),1)
        self.assertIn('leg=2 request=1',s);self.assertIn('leg=3 request=2',s)
        self.assertIn('BGEN 0 2 PRIVATE_PROMPT',e.proc.stdin.getvalue())
    def test_trace_off_preserves_wire_without_output(self):
        e=self.engine(False);out=io.StringIO()
        with contextlib.redirect_stderr(out):e._slot_trace_begin();e._send('GEN 2 4,5');e._slot_trace_first()
        self.assertEqual(out.getvalue(),'');self.assertEqual(e.proc.stdin.getvalue(),'GEN 2 4,5\n')
    def test_process_generation_restarts_leg_counter(self):
        e=self.engine(True);out=io.StringIO()
        with contextlib.redirect_stderr(out):
            e._slot_trace_begin();e._send('GEN 2 1');e.proc.pid=124;e._send('GEN 2 1')
        self.assertIn('pid=124 leg=1',out.getvalue())
    def test_failed_write_is_not_counted_as_a_native_leg(self):
        from serve.server import EngineDied
        e=self.engine(True);e.exit_code=lambda: 1
        e.proc.stdin=mock.Mock();e.proc.stdin.write.side_effect=BrokenPipeError()
        with contextlib.redirect_stderr(io.StringIO()) as out:
            with self.assertRaises(EngineDied):e._send('GEN 2 secret')
        self.assertEqual(out.getvalue(),'');self.assertEqual(e.slot_trace_legs,{})
