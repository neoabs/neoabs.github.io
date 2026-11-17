import unittest

import pandas as pd

from icx.silicon_data import build_silicon_index


def _quote(**overrides) -> dict:
    row = {
        "date": "2026-05-17",
        "provider_id": "aws",
        "provider_country": "US",
        "gpu_model": "nvidia-h100",
        "billing_type": "ON_DEMAND",
        "hourly_per_gpu": 4.0,
        "gpu_count": 8,
        "availability": "AVAILABLE",
        "interconnect": "SXM",
    }
    row.update(overrides)
    return row


class SiliconDataTests(unittest.TestCase):
    def test_weights_geographies_and_filters(self) -> None:
        quotes = pd.DataFrame(
            [
                _quote(provider_id="aws", hourly_per_gpu=4.0, gpu_count=8, interconnect="SXM"),
                _quote(provider_id="ace", provider_country="IN", hourly_per_gpu=2.0, gpu_count=2, interconnect="SXM"),
                _quote(provider_id="aws", hourly_per_gpu=9.0, gpu_count=1, interconnect="PCIe"),
                _quote(provider_id="coreweave", availability="WAITLIST", hourly_per_gpu=5.0, gpu_count=4),
                _quote(provider_id="lambda", availability="UNKNOWN", hourly_per_gpu=3.0, gpu_count=4, interconnect=None),
                _quote(provider_id="runpod", availability="UNAVAILABLE", hourly_per_gpu=1.0, gpu_count=100),
                _quote(billing_type="CUSTOM", hourly_per_gpu=1.0, gpu_count=100),
            ]
        )
        index = build_silicon_index(quotes)
        sxm = index[
            (index.geography == "GLOBAL")
            & (index.interconnect == "SXM")
            & (index.availability == "AVAILABLE")
            & (index.billing == "ON_DEMAND")
        ]
        self.assertEqual(len(sxm), 1)
        # 4*8 + 2*2 over 10 listed GPUs.
        self.assertAlmostEqual(sxm.iloc[0].index_value, 3.6)
        self.assertEqual(int(sxm.iloc[0].n_providers), 2)

        pcie = index[(index.geography == "GLOBAL") & (index.interconnect == "PCIe")]
        self.assertEqual(len(pcie), 1)
        self.assertAlmostEqual(pcie.iloc[0].index_value, 9.0)

        unknown = index[(index.geography == "GLOBAL") & (index.interconnect == "UNKNOWN")]
        self.assertEqual(set(unknown.availability), {"UNKNOWN"})

        waitlist = index[(index.geography == "AMER") & (index.availability == "WAITLIST")]
        self.assertEqual(len(waitlist), 1)
        self.assertAlmostEqual(waitlist.iloc[0].index_value, 5.0)

        self.assertTrue(index[index.availability == "UNAVAILABLE"].empty)
        self.assertTrue(index[index.billing == "CUSTOM"].empty)
        us = index[(index.geography == "US") & (index.interconnect == "SXM") & (index.availability == "AVAILABLE")]
        self.assertEqual(len(us), 1)
        apac = index[(index.geography == "APAC") & (index.interconnect == "SXM") & (index.availability == "AVAILABLE")]
        self.assertAlmostEqual(apac.iloc[0].index_value, 2.0)
