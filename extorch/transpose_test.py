import torch

from .transpose import Transpose


def test_transpose():
    """Conv1dEx should support causal convolution w/o stride"""

    with torch.no_grad():
        # (B, T, Feat) -> (B, Feat, T)
        i = torch.tensor([
            [ # B=1
          # Feat= 1    2    3
                [1.1, 1.2, 1.3], # t=1
                [2.1, 2.2, 2.3], # t=2
                [3.1, 3.2, 3.3], # t=3
            ],
            [ # B=2
          # Feat= 1     2     3
                [21.1, 21.2, 21.3], # t=1
                [22.1, 22.2, 22.3], # t=2
                [23.1, 23.2, 23.3], # t=3
            ]
        ])
        opt_gt = torch.tensor([
            [ # B=1
          # T=    1    2    3
                [1.1, 2.1, 3.1], # Feat=1
                [1.2, 2.2, 3.2], # Feat=2
                [1.3, 2.3, 3.3], # Feat=3
            ],
            [ # B=2
          # T=    1     2     3
                [21.1, 22.1, 23.1], # Feat=1
                [21.2, 22.2, 23.2], # Feat=2
                [21.3, 22.3, 23.3], # Feat=3
            ]
        ])
        # Test
        opt = Transpose(1,2)(i)
        assert torch.equal(opt, opt_gt), f"{opt} vs {opt_gt}"
