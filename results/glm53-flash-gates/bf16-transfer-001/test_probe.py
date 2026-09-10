"""Reject framing errors before reading an unexpectedly large response."""
import importlib.util,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('transport',Path(__file__).with_name('probe.py'));p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
class Headers(unittest.TestCase):
    def test_exact_partial_response(self):p.check_headers(206,{'Content-Range':'bytes 10-19/100','Content-Length':'10'},10,19,100)
    def test_ignore_range_rejected(self):
        with self.assertRaisesRegex(ValueError,'206'):p.check_headers(200,{'Content-Length':'100'},10,19,100)
    def test_wrong_framing_rejected(self):
        for h in [{'Content-Range':'bytes 10-20/100','Content-Length':'10'},{'Content-Range':'bytes 10-19/101','Content-Length':'10'},{'Content-Range':'bytes 10-19/100','Content-Length':'100'},{'Content-Range':'bytes 10-19/100','Content-Length':'10','Content-Encoding':'gzip'},{}]:
            with self.subTest(h=h),self.assertRaises(ValueError):p.check_headers(206,h,10,19,100)
    def test_missing_arms_rejected(self):
        with self.assertRaisesRegex(ValueError,'four'):p.score([])

class RawHeaders(unittest.TestCase):
    def response(self,headers,body):
        import io,http.client
        class Socket:
            def makefile(self,*args):return io.BytesIO(b'HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 0-9/100\r\n'+headers+b'\r\n'+body)
        r=http.client.HTTPResponse(Socket());r.begin();return r
    def test_conflicting_lengths_rejected(self):
        r=self.response(b'Content-Length: 10\r\nContent-Length: 999\r\n',b'0123456789')
        with self.assertRaises(ValueError):p.check_headers(r.status,r.headers,0,9,100)
    def test_chunked_and_length_rejected(self):
        r=self.response(b'Transfer-Encoding: chunked\r\nContent-Length: 10\r\n',b'a\r\n0123456789\r\n0\r\n\r\n')
        with self.assertRaises(ValueError):p.check_headers(r.status,r.headers,0,9,100)
if __name__=='__main__':unittest.main()
