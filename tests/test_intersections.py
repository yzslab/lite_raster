"""CUDA parity and safety checks for the public single-view intersection operators."""
import unittest

import torch
from lite_raster import Mode, isect_encode, isect_encode_precise, project, rasterize


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TileIntersections(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.n = 385
        w, h = 125, 111
        self.pre = (torch.eye(4, device="cuda")[None],
                    torch.tensor([[[100., 0, 60.], [0, 100., 55.], [0., 0., 1.]]], device="cuda"),
                    (w, h), torch.zeros(1, 3, device="cuda"))
        self.means = torch.randn(self.n, 3, device="cuda") * .5
        self.means[:, 2] = 2
        self.means[:9, 2] = -1
        self.scales = torch.rand_like(self.means) * .5 + .01
        self.scales[16] = 5
        self.quats = torch.randn(self.n, 4, device="cuda")
        self.opacity = torch.rand(self.n, device="cuda")
        self.opacity[10:15] = 0
        self.colors = torch.rand_like(self.means)

    def render(self, aa, n, tile=16, *, chained=None, precise=True):
        means, scales, quats, opacity = [t[:n].detach().requires_grad_() for t in
                                         (self.means, self.scales, self.quats, self.opacity)]
        options = dict(compensate=aa, normals=True, planes=True)
        if chained:
            full, plane, normal, isects = project(self.pre, means, scales, quats,
                                                  isect_opacities=opacity if precise else None,
                                                  tile_size=tile, chained_isect=chained, **options)
        else:
            full, plane, normal = project(self.pre, means, scales, quats, **options)
        radii, xy, z, conic = full[:4]
        alpha = opacity * full[7][0] if aa else opacity
        if not chained:
            encoder = isect_encode_precise if precise else isect_encode
            isects = encoder(self.pre, full, alpha, tile)
        if tile != 16:
            return isects, (), ()
        result = rasterize(xy[0], conic[0], alpha, self.colors[:n], z[0], plane, normal, None,
                           self.pre[1][0], isects[3], isects[2], self.pre[2],
                           Mode.EXPECTED | Mode.RADE | Mode.NORMAL | Mode.ALPHA)
        images = (result.render, result.exp_depth, result.normal, result.alpha)
        loss = sum(im.square().mean() for im in images)
        grads = torch.autograd.grad(loss, (means, scales, quats, opacity))
        return isects, images, grads

    def test_equal_depth_order_and_offsets(self):
        for kind in ("precise", "bbox"):
            with self.subTest(kind=kind):
                (counts, keys, ids, offsets), _, _ = self.render(True, self.n, 8, chained=kind, precise=kind == "precise")
                self.assertEqual(counts.sum().item(), len(keys))
                same = keys[1:] == keys[:-1]
                self.assertTrue(same.any())
                self.assertTrue((ids[1:][same] > ids[:-1][same]).all())
                tiles = torch.arange(offsets.numel(), device=offsets.device, dtype=torch.int64)
                torch.testing.assert_close(offsets.flatten().long(), torch.searchsorted(keys >> 32, tiles))

    def test_pre_aa_opacity_shorthand_returns_precise_list(self):
        args = self.pre, self.means, self.scales, self.quats
        full, _, _, isects = project(*args, compensate=True, isect_opacities=self.opacity)
        ref = isect_encode_precise(self.pre, full, self.opacity * full[7][0])
        self.assertEqual(len(isects), len(ref))
        for actual, expected in zip(isects, ref):
            self.assertTrue(torch.equal(actual, expected))

    def test_chained_matches_separate_on_current_stream(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for kind in ("precise", "bbox"):
                for aa, n, tile in ((True, self.n, 16), (False, self.n, 16), (False, self.n, 8),
                                    (True, 0, 16), (True, 9, 16)):
                    with self.subTest(kind=kind, aa=aa, n=n, tile=tile):
                        args = (aa, n, tile)
                        expected = self.render(*args, precise=kind == "precise")
                        actual = self.render(*args, chained=kind, precise=kind == "precise")
                        for left, right in zip(expected[0] + expected[1], actual[0] + actual[1]):
                            self.assertTrue(torch.equal(left, right))
                        for left, right in zip(expected[2], actual[2]):
                            torch.testing.assert_close(left, right, rtol=2e-4, atol=2e-6)

    def test_chained_precise_requires_matching_opacities(self):
        args = self.pre, self.means, self.scales, self.quats
        with self.assertRaisesRegex(ValueError, "isect_opacities"):
            project(*args, chained_isect="precise")
        with self.assertRaisesRegex(ValueError, "one FP32 opacity"):
            project(*args, chained_isect="precise", isect_opacities=self.opacity[:-1])

    def test_against_gsplat(self):
        try:
            from gsplat.cuda._wrapper import isect_tiles, isect_offset_encode
            from gsplat.cuda.isect_tiles_speedy import isect_tiles_speedy
        except ImportError:
            self.skipTest("optional gsplat reference unavailable")
        full, _, _ = project(self.pre, self.means, self.scales, self.quats, compensate=True)
        radii, xy, z, conic = full[:4]
        opacity = self.opacity * full[7]
        for size in (8, 16, 32):
            w, h = self.pre[2]
            grid = (size, (w+size-1)//size, (h+size-1)//size)
            for precise in (False, True):
                fn = isect_encode_precise if precise else isect_encode
                actual = fn(preprocessed_camera=self.pre, projection_results=full, opacities=opacity, tile_size=size)
                if precise:
                    ref = isect_tiles_speedy(xy, radii, z, conic, opacity, *grid, n_cameras=1)
                else:
                    ref = isect_tiles(xy, radii, z, *grid, n_cameras=1)
                offsets = isect_offset_encode(ref[1], 1, *grid[1:])
                for x, y in zip(actual, (*ref, offsets)):
                    self.assertTrue(torch.equal(x, y), (size, precise))
