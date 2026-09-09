"""Independent indexer rank, planar-cache and pool/tail reference checks."""
import importlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))


class IndexerFixtureTests(unittest.TestCase):
    def setUp(self): self.api = importlib.import_module('glm53_indexer_fixture')

    def test_ranks_and_physical_reservations_distinguish_requests(self):
        import numpy as np
        a = self.api
        self.assertEqual(int(a.ranks(7, 2)[123]), 17004)
        tops = []
        for request in range(4):
            np.testing.assert_array_equal(np.sort(a.ranks(7, request)), np.arange(65536))
            tops.append(set(a.top_history(7, request, 65535)))
        self.assertEqual(len({frozenset(x) for x in tops}), 4)
        blocks = a.physical_blocks(7)
        self.assertEqual(sorted(x for row in blocks for x in row), list(range(1,145)))
        self.assertTrue(all(len(row) == 36 for row in blocks))
        for bad in (True, -1, 2**64):
            with self.assertRaises(ValueError): a.case_order(bad)

    def good_indices(self, seed, case):
        import numpy as np
        a = self.api; cfg = a.case_config(seed, case)
        result = np.full((len(cfg['positions']), 2048), -1, dtype='<i4')
        tops = {(int(r),int(n)):a.top_history(seed,int(r),int(n)) for r,n in set(zip(cfg['row_requests'],cfg['history_counts']))}
        for i, (request, position, old_count) in enumerate(zip(cfg['row_requests'], cfg['positions'], cfg['history_counts'])):
            pools = tops[(int(request),int(old_count))]
            # Omit any one of the 512 and permute group order: native top-k is unsorted.
            chosen = np.roll(pools, i % 512)[:511][::-1]
            result[i, :2044] = (chosen[:, None]*4 + np.arange(4)).reshape(-1)
            n = (int(position)+1)%4
            result[i,2044:2044+n] = np.arange(int(position)//4*4, int(position)//4*4+n)
        return result

    def test_unsorted_top512_subset_tail_and_mutations(self):
        import numpy as np
        a = self.api; seed = 7; case = 'decode-4'; data = self.good_indices(seed, case)
        self.assertEqual(a.score_indices(data, seed, case)['rows'], 4)
        np.testing.assert_array_equal(data[:,2044:], [[262140,-1,-1,-1], [262140,262141,-1,-1], [262140,262141,262142,-1], [-1,-1,-1,-1]])
        cfg = a.case_config(seed, case)
        for mutation in ('duplicate', 'wrong_request', 'padding', 'future', 'missing', 'dtype'):
            changed = data.copy()
            if mutation == 'duplicate': changed[0,4:8] = changed[0,:4]
            if mutation == 'wrong_request': changed[0,:2044] = changed[1,:2044]
            if mutation == 'padding': changed[3,2044:] = [262140,262141,262142,262143]
            if mutation == 'future': changed[0,:4] = [262144,262145,262146,262147]
            if mutation == 'missing': changed = changed[:3]
            if mutation == 'dtype': changed = changed.astype('float32')
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): a.score_indices(changed, seed, case)
        self.assertEqual(len(cfg['positions']), 4)

    def test_planar_cache_and_exact_phase_three_compression(self):
        import numpy as np
        a = self.api; seed = 7; cfg = a.case_config(seed, 'decode-4')
        initial = a.cache_and_tail(seed, 'decode-4', final=False)
        final = a.cache_and_tail(seed, 'decode-4', final=True)
        self.assertEqual(initial[0].shape, (4930,8448))
        self.assertEqual(initial[1].shape, (145,2,4,128))
        np.testing.assert_array_equal(initial[0][0], np.full(8448,0x55,dtype='uint8'))
        blocks = a.physical_blocks(seed)
        request = int(cfg['row_requests'][0]); page = blocks[request][0]*34
        np.testing.assert_array_equal(initial[0][page,:128], np.array([0x38]+[0]*127,dtype='uint8'))
        scale = np.frombuffer(initial[0][page,8192:].tobytes(), dtype='<f4')[0]
        self.assertEqual(float(scale), (int(a.ranks(seed,request)[0])+1)/65536)
        request = int(cfg['row_requests'][3]); page = blocks[request][30]*34 + 3
        # Pool65535 occupies offset63 in the fourth pooled page of manager block30.
        value, expected_scale = a.compressed_key(request)
        np.testing.assert_array_equal(final[0][page,63*128:64*128], value)
        self.assertEqual(float(final[0][page,8192:].view('<f4')[63]), expected_scale)
        self.assertEqual(int(np.count_nonzero(value)), 1)
        for i, request in enumerate(cfg['row_requests']):
            tail_block = blocks[int(request)][31]
            for phase in range(4):
                bits = a.bf16_bits(a.raw_value(int(request),phase)) if phase <= i else 0x4050
                self.assertTrue(np.all(final[1][tail_block,0,phase] == bits))
                self.assertTrue(np.all(final[1][tail_block,1,phase] == (0 if phase <= i else 0x4050)))

    def test_prefill_and_valid_logits_contract(self):
        import numpy as np
        a = self.api
        for case in ('prefill-1','prefill-4'):
            indices = self.good_indices(7,case)
            self.assertEqual(a.score_indices(indices,7,case)['rows'],2048)
        for request in range(4):
            old = 65408; values = a.valid_logits(7,request,old,65536)
            np.testing.assert_array_equal(values[:old], ((a.ranks(7,request)[:old]+1)/65536).astype('<f4'))
            vector,scale = a.compressed_key(request)
            self.assertTrue(np.all(values[old:] == a.fp8_value(int(vector[0]))*scale))
            self.assertLess(float(values[old]), float(np.partition(values[:old],-512)[-512]))
            self.assertTrue(np.isfinite(values).all())
        before, _ = a.cache_and_tail(7,'decode-4',final=False)
        after, _ = a.cache_and_tail(7,'decode-4',final=True)
        request = int(a.case_config(7,'decode-4')['row_requests'][3])
        page = a.physical_blocks(7)[request][30]*34+3
        allowed = np.zeros(before.shape,dtype='bool'); allowed[page,8064:8192] = True; allowed[page,8444:8448] = True
        np.testing.assert_array_equal(before[~allowed],after[~allowed])

    def test_bf16_and_fp8_rounding_independent_known_values(self):
        a = self.api
        self.assertEqual(a.bf16_bits(3.25), 0x4050)
        self.assertEqual(a.fp8_positive(1.), 0x38)
        self.assertEqual(a.fp8_positive(448.), 0x7e)
        self.assertEqual(a.fp8_positive(1.0625), 0x38)
        self.assertEqual(a.fp8_positive(1.1875), 0x3a)
        with self.assertRaises(ValueError): a.fp8_positive(float('nan'))


if __name__ == '__main__': unittest.main()
