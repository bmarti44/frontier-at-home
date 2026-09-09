"""Byte extraction must retain overlaps and reject corrupt or incomplete sources."""
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('weight_download',Path(__file__).resolve().parents[1]/'46_prepare_glm53_weights.py')
api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)


class SmallReads(io.BytesIO):
    def read(self,size=-1):return super().read(min(size,7))


class WeightDownloadTests(unittest.TestCase):
    def test_dense_range_two_reads_and_exact_content_range(self):
        source=bytes(range(128));segments=[(0,8,0),(5,36,8),(113,128,39)]
        class Response(io.BytesIO):
            status=206
            headers={'Content-Range':'bytes 0-127/128'}
        with tempfile.TemporaryFile() as output:
            with patch.object(api.urllib.request,'urlopen',side_effect=lambda *a,**kw:Response(source)) as calls:
                count,ranges=api.download_dense_ranges('https://example.invalid/pinned?x=1',output.fileno(),segments,len(source))
                self.assertEqual(calls.call_count,2);self.assertEqual(count,54)
                output.seek(0);self.assertEqual(output.read(),source[:8]+source[5:36]+source[113:])
            with patch.object(api.urllib.request,'urlopen',side_effect=[Response(source),Response(source[:-1]+b'x')]):
                with self.assertRaisesRegex(ValueError,'reads differ'):
                    api.download_dense_ranges('https://example.invalid/pinned?x=1',output.fileno(),segments,len(source))
            Response.headers={'Content-Range':'bytes 1-128/129'}
            with patch.object(api.urllib.request,'urlopen',return_value=Response(source)):
                with self.assertRaisesRegex(ValueError,'exact pinned'):
                    api.download_dense_ranges('https://example.invalid/pinned?x=1',output.fileno(),segments,len(source))

    def test_exact_overlapping_and_split_source_ranges(self):
        source=bytes(range(128)); segments=[(0,8,0),(5,36,8),(5,36,39),(113,128,70)]
        with tempfile.TemporaryFile() as output:
            count=api.stream_selected(SmallReads(source),output.fileno(),segments,len(source),hashlib.sha256(source).hexdigest())
            output.seek(0);result=output.read()
            expected=source[:8]+source[5:36]*2+source[113:]
            self.assertEqual(result,expected);self.assertEqual(count,len(expected))
            self.assertEqual(api.selected_digest(output.fileno(),segments),hashlib.sha256(expected).hexdigest())

    def test_corrupt_short_extra_or_invalid_ranges_rejected(self):
        source=b'original bytes';expected=hashlib.sha256(source).hexdigest()
        with tempfile.TemporaryFile() as output:
            for data in (b'corrupt! bytes',source[:-1],source+b'x'):
                with self.subTest(data=data),self.assertRaises(ValueError):
                    api.stream_selected(SmallReads(data),output.fileno(),[(0,3,0)],len(source),expected)
            for segment in ((-1,3,0),(1,20,0),(1,1,0),(0,3,-1)):
                with self.subTest(segment=segment),self.assertRaises(ValueError):
                    api.stream_selected(SmallReads(source),output.fileno(),[segment],len(source),expected)


if __name__=='__main__':unittest.main()
